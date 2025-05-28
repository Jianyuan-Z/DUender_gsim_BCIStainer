import torch.nn as nn

from .utils import *
from .layers import *
from copy import deepcopy
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def define_G(configs):

    if configs.name == 'basic':
        net = BCIStainerBasic(**configs.params)
    elif configs.name == 'cahr':
        net = BCIStainerCAHR(**configs.params)
    elif configs.name == 'mamba_gsim':
        net = BCIStainerBasic_mamba_gsim(**configs.params)
    else:
        raise NotImplementedError(f'unknown G model name {configs.name}')

    init_weights(net, **configs.init)
    return net


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class GSIM(nn.Module):
    def __init__(self, loacl_channels, global_channels):
        super(GSIM, self).__init__()

        # Local branch
        self.local_conv = nn.Conv2d(loacl_channels, loacl_channels, kernel_size=1)
        self.bn_local = nn.BatchNorm2d(loacl_channels)

        # Global branch
        self.global_conv1 = nn.Conv2d(global_channels, loacl_channels, kernel_size=1)
        self.bn_global1 = nn.BatchNorm2d(loacl_channels)
        self.sigmoid = nn.Sigmoid()
        self.global_conv2 = nn.Conv2d(global_channels, loacl_channels, kernel_size=1)
        self.bn_global2 = nn.BatchNorm2d(loacl_channels)

    def forward(self, global_features, local):
        # Process the local features
        local_out = self.bn_local(self.local_conv(local))

        # Process the global features
        global_out_1 = self.bn_global1(self.global_conv1(global_features))
        global_out_1 = self.sigmoid(global_out_1)

        y = local_out * global_out_1

        global_out_2 = self.bn_global2(self.global_conv2(global_features))
        y = y + global_out_2

        return y

    
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x



class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)

        return x


class BCIStainerBasic_mamba_gsim(nn.Module):

    def __init__(self,
                 full_size=1024,
                 input_channels=3,
                 output_channels=3,
                 init_channels=32,
                 levels=4,
                 encoder1_blocks=3,
                 style_type='mod',
                 style_linear=True,
                 style_blocks=9,
                 norm_type='batch',
                 dropout=0.2,
                 output_lowres=True,
                 attention=False,
                 drop_path=0.,
                 attn_drop=0.,
                 d_state=16,
                 norm_layer_1=nn.LayerNorm,
                 img_size=1024, patch_size=4, in_chans=3, embed_dim=48, num_heads=4, window_size=8, shift_size=0,
                 norm_layer=None
                 ):
        super(BCIStainerBasic_mamba_gsim, self).__init__()
        self.patch_embed1 = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer_1
        )
        self.vss_block1 = VSSBlock(
            hidden_dim=embed_dim,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        # PatchMerging 层
        self.patch_merge1 = PatchMerging2D(
            dim=embed_dim,
            norm_layer=norm_layer_1
        )
        self.vss_block2 = VSSBlock(
            hidden_dim=96,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        self.patch_merge2 = PatchMerging2D(
            dim=96,
            norm_layer=norm_layer_1
        )
        self.vss_block3 = VSSBlock(
            hidden_dim=192,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        self.patch_merge3 = PatchMerging2D(
            dim=192,
            norm_layer=norm_layer_1
        )
        self.vss_block4 = VSSBlock(
            hidden_dim=384,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        self.patch_merge4 = PatchMerging2D(
            dim=384,
            norm_layer=norm_layer_1
        )
        self.vss_block5 = VSSBlock(
            hidden_dim=768,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        self.patch_merge5 = PatchMerging2D(
            dim=768,
            norm_layer=norm_layer_1
        )
        self.vss_block6 = VSSBlock(
            hidden_dim=768 * 2,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer_1,
            attn_drop_rate=attn_drop,
            d_state=d_state,
        )
        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        self.inconv = ConvNormAct(
            in_dims=input_channels, out_dims=init_channels,
            conv_type='conv2d', kernel_size=7, stride=1,
            padding=3, bias=use_bias, norm_layer=norm_layer,
            sampling='none', attention=False
        )

        encoder1 = []
        for i in range(encoder1_blocks):
            mult = 2 ** i
            in_dims = init_channels * mult
            out_dims = init_channels * mult * 2
            encoder1.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.encoder1 = nn.Sequential(*encoder1)

        style_dims = conv_dims = out_dims
        total_encoder_blocks = int(np.log2(full_size / 8))
        num_encoder2_blocks = total_encoder_blocks - encoder1_blocks

        encoder2 = []
        for i in range(num_encoder2_blocks):
            encoder2.append(
                ConvNormAct(
                    in_dims=style_dims, out_dims=style_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.encoder2 = nn.Sequential(*encoder2)

        classify_head = []
        if dropout > 0:
            classify_head.append(nn.Dropout(dropout))
        classify_head.append(nn.Linear(style_dims, levels))
        self.classify_head = nn.Sequential(*classify_head)

        assert style_type in ['ada', 'mod', 'none']
        self.style_type = style_type

        decoder1 = []
        for i in range(style_blocks):
            if self.style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    attention=attention
                )
            elif self.style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    style_linear=style_linear,
                    attention=attention
                )
            else:  # self.style_type == 'none'
                layer = ResnetBlock(
                    conv_dims,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                    attention=attention
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=0
                ),
                nn.Tanh()
            )

        decoder2 = []
        for i in range(encoder1_blocks):
            mult = 2 ** (encoder1_blocks - i)
            in_dims = init_channels * mult
            out_dims = int(init_channels * mult / 2)
            decoder2.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='convTranspose2d',
                    kernel_size=3, stride=2, padding=1,
                    bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.decoder2 = nn.Sequential(*decoder2)

        self.highres_outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=0
            ),
            nn.Tanh()
        )
        self.gsim1 = GSIM(256, 96)
        self.gsim2 = GSIM(256, 1536)

    def forward(self, he):
        B, C, H, W = he.shape
        he_in = self.inconv(he)
        enc1 = self.encoder1(he_in)
        x = self.patch_embed1(he)
        x = self.vss_block1(x)
        x = self.vss_block1(x)
        x = self.patch_merge1(x)
        x = self.vss_block2(x)
        x = self.vss_block2(x)
        x1 = x.view(B, 128, 128, 96)
        x1 = x1.view(B, 96, 128, 128)

        # conv_layer_correct = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0).to('cuda')
        # mamba_tensor = conv_layer_correct(x1)
        # fused_tensor_elementwise_add = enc1 + mamba_tensor
        fused_tensor_elementwise_add = self.gsim1(x1, enc1)

        x = self.patch_merge2(x)
        x = self.vss_block3(x)
        x = self.vss_block3(x)
        x = self.patch_merge3(x)
        x = self.vss_block4(x)
        x = self.vss_block4(x)
        x = self.patch_merge4(x)
        x = self.vss_block5(x)
        x = self.vss_block5(x)
        x = self.patch_merge5(x)
        x = self.vss_block6(x)
        x = self.vss_block6(x)

        x2 = x.view(B, 8, 8, 1536)
        x2 = x2.view(B, 1536, 8, 8)

        # conv_layer_correct_2 = nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0).to('cuda')
        # mamba_tensor_2 = conv_layer_correct_2(x2)

        style_before = self.encoder2(fused_tensor_elementwise_add)
        global_pool = nn.AdaptiveAvgPool2d(1).to('cuda')
        flatten = nn.Flatten(start_dim=1).to('cuda')

        # fused_tensor_elementwise_add_2 = style_before + mamba_tensor_2
        fused_tensor_elementwise_add_2 = self.gsim2(x2, style_before)

        style = global_pool(fused_tensor_elementwise_add_2)  # 应用全局平均池化
        style = flatten(style)  # 展平张量
        level = self.classify_head(style)
        if self.style_type == 'none':
            dec1 = self.decoder1(fused_tensor_elementwise_add)
        else:
            dec1, _ = self.decoder1([fused_tensor_elementwise_add, style])

        dec2 = self.decoder2(dec1)
        ihc_hr = self.highres_outconv(dec2)

        if self.output_lowres:
            ihc_lr = self.lowres_outconv(dec1)
            return ihc_hr, ihc_lr, level
        else:
            return ihc_hr, level



class BCIStainerBasic(nn.Module):

    def __init__(self,
        full_size=1024,
        input_channels=3,
        output_channels=3,
        init_channels=32,
        levels=4,
        encoder1_blocks=3,
        style_type='mod',
        style_linear=True,
        style_blocks=9,
        norm_type='batch',
        dropout=0.2,
        output_lowres=True,
        attention=False
    ):
        super(BCIStainerBasic, self).__init__()

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        self.inconv = ConvNormAct(
            in_dims=input_channels, out_dims=init_channels,
            conv_type='conv2d', kernel_size=7, stride=1,
            padding=3, bias=use_bias, norm_layer=norm_layer,
            sampling='none', attention=False
        )

        encoder1 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** i
            in_dims  = init_channels * mult
            out_dims = init_channels * mult * 2
            encoder1.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.encoder1 = nn.Sequential(*encoder1)

        style_dims = conv_dims = out_dims
        total_encoder_blocks = int(np.log2(full_size / 8))
        num_encoder2_blocks = total_encoder_blocks - encoder1_blocks

        encoder2 = []
        for i in range(num_encoder2_blocks):
            encoder2.append(
                ConvNormAct(
                    in_dims=style_dims, out_dims=style_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        encoder2.append(nn.AdaptiveAvgPool2d(1))
        encoder2.append(nn.Flatten(1))
        self.encoder2 = nn.Sequential(*encoder2)

        classify_head = []
        if dropout > 0:
            classify_head.append(nn.Dropout(dropout))
        classify_head.append(nn.Linear(style_dims, levels))
        self.classify_head = nn.Sequential(*classify_head)

        assert style_type in ['ada', 'mod', 'none']
        self.style_type = style_type

        decoder1 = []
        for i in range(style_blocks):
            if self.style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    attention=attention
                )
            elif self.style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    style_linear=style_linear,
                    attention=attention
                )
            else:  # self.style_type == 'none'
                layer = ResnetBlock(
                    conv_dims,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                    attention=attention
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=0
                ),
                nn.Tanh()
            )

        decoder2 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** (encoder1_blocks - i)
            in_dims  = init_channels * mult
            out_dims = int(init_channels * mult / 2)
            decoder2.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='convTranspose2d',
                    kernel_size=3, stride=2, padding=1,
                    bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.decoder2 = nn.Sequential(*decoder2)

        self.highres_outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=0
            ),
            nn.Tanh()
        )

    def forward(self, he):

        he_in = self.inconv(he)
        enc1 = self.encoder1(he_in)
        style = self.encoder2(enc1)
        level = self.classify_head(style)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            dec1, _ = self.decoder1([enc1, style])

        dec2 = self.decoder2(dec1)
        ihc_hr = self.highres_outconv(dec2)

        if self.output_lowres:
            ihc_lr = self.lowres_outconv(dec1)
            return ihc_hr, ihc_lr, level
        else:
            return ihc_hr, level


class BCIStainerCAHR(nn.Module):

    def __init__(self,
        full_size=1024,
        crop_size=512,
        input_channels=3,
        output_channels=3,
        init_channels=64,
        levels=4,
        encoder1_blocks=2,
        style_type='mod',
        style_linear=True,
        style_blocks=9,
        norm_type='batch',
        dropout=0.2,
        output_lowres=True,
        mask_dec_input='dec1',
        attention=False
    ):
        super(BCIStainerCAHR, self).__init__()

        self.full_size = full_size
        self.crop_size = crop_size

        assert norm_type in ['batch', 'instance', 'none']
        norm_layer = get_norm_layer(norm_type=norm_type)
        use_bias = False if norm_type == 'batch' else True

        self.inconv = ConvNormAct(
            in_dims=input_channels, out_dims=init_channels,
            conv_type='conv2d', kernel_size=7, stride=1,
            padding=3, bias=use_bias, norm_layer=norm_layer,
            sampling='none', attention=False
        )

        encoder1 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** i
            in_dims  = init_channels * mult
            out_dims = init_channels * mult * 2
            encoder1.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.encoder1 = nn.Sequential(*encoder1)

        style_dims = conv_dims = out_dims
        total_encoder_blocks = int(np.log2(full_size / 8))
        num_encoder2_blocks = total_encoder_blocks - encoder1_blocks

        encoder2 = []
        for i in range(num_encoder2_blocks):
            encoder2.append(
                ConvNormAct(
                    in_dims=style_dims, out_dims=style_dims,
                    conv_type='conv2d', kernel_size=3, stride=2,
                    padding=1, bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        encoder2.append(nn.AdaptiveAvgPool2d(1))
        encoder2.append(nn.Flatten(1))
        self.encoder2 = nn.Sequential(*encoder2)

        classify_head = []
        if dropout > 0:
            classify_head.append(nn.Dropout(dropout))
        classify_head.append(nn.Linear(style_dims, levels))
        self.classify_head = nn.Sequential(*classify_head)

        assert style_type in ['ada', 'mod', 'none']
        self.style_type = style_type

        decoder1 = []
        for i in range(style_blocks):
            if self.style_type == 'ada':
                layer = ResnetAdaBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    attention=attention
                )
            elif self.style_type == 'mod':
                layer = ResnetModBlock(
                    style_dims, conv_dims,
                    use_bias=use_bias,
                    style_linear=style_linear,
                    attention=attention
                )
            else:  # self.style_type == 'none'
                layer = ResnetBlock(
                    conv_dims,
                    norm_layer=norm_layer,
                    use_bias=use_bias,
                    attention=attention
                )
            decoder1.append(layer)
        self.decoder1 = nn.Sequential(*decoder1)

        self.output_lowres = output_lowres
        if self.output_lowres:
            self.lowres_outconv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    conv_dims, output_channels,
                    kernel_size=3, padding=0
                ),
                nn.Tanh()
            )

        decoder2 = []
        for i in range(encoder1_blocks):
            mult     = 2 ** (encoder1_blocks - i)
            in_dims  = init_channels * mult
            out_dims = int(init_channels * mult / 2)
            decoder2.append(
                ConvNormAct(
                    in_dims=in_dims, out_dims=out_dims,
                    conv_type='convTranspose2d',
                    kernel_size=3, stride=2, padding=1,
                    bias=use_bias, norm_layer=norm_layer,
                    sampling='none', attention=False
                )
            )
        self.decoder2 = nn.Sequential(*decoder2)

        self.highres_outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, output_channels,
                kernel_size=7, padding=0
            ),
            nn.Tanh()
        )

        assert mask_dec_input in ['dec1', 'enc1'], \
            f'mask_dec_input {mask_dec_input} is invalid'
        self.mask_dec_input = mask_dec_input
        self.mask_decoder = deepcopy(self.decoder2)
        self.mask_outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                init_channels, 1,
                kernel_size=7, padding=0
            ),
            nn.Sigmoid()
        )

    def _forward_full(self, he):

        outputs_dict = {}

        he_in = self.inconv(he)
        enc1 = self.encoder1(he_in)
        style = self.encoder2(enc1)
        level = self.classify_head(style)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            outputs_dict['style'] = style
            dec1, _ = self.decoder1([enc1, style])

        dec2 = self.decoder2(dec1)
        ihc_full = self.highres_outconv(dec2)

        if self.output_lowres:
            ihc_lr = self.lowres_outconv(dec1)
            outputs_dict['ihc_lr'] = ihc_lr
        
        outputs_dict['level']    = level
        outputs_dict['ihc_full'] = ihc_full
        return outputs_dict

    def _forward_crop(self, he_crop, style):

        he_in = self.inconv(he_crop)
        enc1 = self.encoder1(he_in)

        if self.style_type == 'none':
            dec1 = self.decoder1(enc1)
        else:
            if style.size(0) != he_crop.size(0):
                style = style.repeat(he_crop.size(0), 1)
            dec1, _ = self.decoder1([enc1, style])
        
        dec2 = self.decoder2(dec1)
        ihc_crop = self.highres_outconv(dec2)

        if self.mask_dec_input == 'dec1':
            mask_dec = self.mask_decoder(dec1)
        elif self.mask_dec_input == 'enc1':
            mask_dec = self.mask_decoder(enc1)
        mask_crop = self.mask_outconv(mask_dec)

        outputs_dict = {
            'ihc_crop': ihc_crop,
            'mask_crop': mask_crop
        }
        return outputs_dict

    def _train_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):

        ihc_hr_list = []
        for i in range(ihc_full.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size
            row_pad = [row1, self.full_size - row2]
            col_pad = [col1, self.full_size - col2]

            ihc_crop_pad  = F.pad(ihc_crop[i], col_pad + row_pad)
            mask_crop_pad = F.pad(mask_crop[i], col_pad + row_pad)

            ihc_full_ = ihc_full[i] * (1 - mask_crop_pad)
            ihc_crop_ = ihc_crop_pad * mask_crop_pad
            ihc_hr_   = ihc_full_ + ihc_crop_
            ihc_hr_list.append(ihc_hr_)

        ihc_hr = torch.stack(ihc_hr_list)
        return ihc_hr

    def _infer_full_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):
        assert ihc_full.size(0) == 1

        ihc_full = ihc_full.squeeze(0)
        ihc_hr   = torch.zeros_like(ihc_full)

        for i in range(ihc_crop.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size
            row_pad = [row1, self.full_size - row2]
            col_pad = [col1, self.full_size - col2]

            ihc_crop_pad  = F.pad(ihc_crop[i], col_pad + row_pad)
            mask_crop_pad = F.pad(mask_crop[i], col_pad + row_pad)

            ihc_full_ = ihc_full * (1 - mask_crop_pad)
            ihc_crop_ = ihc_crop_pad * mask_crop_pad
            ihc_hr_   = ihc_full_ + ihc_crop_
            ihc_hr   += ihc_hr_

        ihc_hr /= ihc_crop.size(0)
        return ihc_hr.unsqueeze(0)

    def _infer_crop_merge(self, ihc_full, ihc_crop, mask_crop, crop_idxs):
        assert ihc_full.size(0) == 1

        ihc_full  = ihc_full.squeeze(0)
        ihc_hr    = torch.zeros_like(ihc_full)
        ihc_count = torch.zeros_like(ihc_full)

        for i in range(ihc_crop.size(0)):
            row1, col1 = crop_idxs[i]
            row2, col2 = crop_idxs[i] + self.crop_size

            ihc_full_crop_ = ihc_full[:, row1:row2, col1:col2]
            ihc_full_crop_ = ihc_full_crop_ * (1 - mask_crop[i])
            ihc_crop_ = ihc_crop[i] * mask_crop[i]
            ihc_hr_   = ihc_full_crop_ + ihc_crop_

            ihc_hr[:, row1:row2, col1:col2]    += ihc_hr_
            ihc_count[:, row1:row2, col1:col2] += 1.0

        ihc_hr /= ihc_count
        return ihc_hr.unsqueeze(0)

    def forward(self, he, he_crop, crop_idxs, mode):
        assert mode in ['train', 'infer_full', 'infer_crop'], \
            f'mode {mode} is invalid'

        full_outputs = self._forward_full(he)
        level    = full_outputs['level']
        ihc_full = full_outputs['ihc_full']
        style    = full_outputs.get('style', None)
        ihc_lr   = full_outputs.get('ihc_lr', None)

        crop_outputs = self._forward_crop(he_crop, style)
        ihc_crop  = crop_outputs['ihc_crop']
        mask_crop = crop_outputs['mask_crop']

        if mode == 'train':
            merge_func = self._train_merge
        elif mode == 'infer_full':
            merge_func = self._infer_full_merge
        elif mode == 'infer_crop':
            merge_func = self._infer_crop_merge

        ihc_hr = merge_func(ihc_full, ihc_crop, mask_crop, crop_idxs)

        if self.output_lowres:
            return ihc_hr, ihc_lr, ihc_crop, level
        else:
            return ihc_hr, ihc_crop, level
