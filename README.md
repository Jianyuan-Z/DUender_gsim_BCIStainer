# DUender_gsim_BCIStainer
DUender_gsim_BCIStainer

<p align="center">
    <br>
    <img src="./assets/图7.png" width=800 />
    <br>
</p>

## 1.Environment
```bash
# using conda
conda create --name bci python=3.8
conda activate bci

# pytorch 1.12.0
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# other packages
pip install -r requirements.txt
```

## 2.Dateset
Download dataset from [BCI page](https://bupt-ai-cz.github.io/BCI/) and put it in [data](./data) directory as folowing file structure:
```
./data
├── test
│   ├── HE
│   ├── IHC
│   └── README.txt
├── train
│   ├── HE
│   ├── IHC
│   └── README.txt
└── val
    ├── HE
    ├── IHC
    └── README.txt
```

## 3.Train
```bash
CUDA_VISIBLE_DEVICES=0          \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file ./configs/stainer_basic_cmp/exp100.yaml \
    --trainer     basic
```

Download pretrained model and put it into above directory:
- BaiduYun: [https://pan.baidu.com/s/1QxZ2zB0CHZKyttXqpS9iDw](https://pan.baidu.com/s/1ygGJho1X2fug458ZvI_SYQ?pwd=fntq)  Code: fntq

## 4.Evaluate
```bash
CUDA_VISIBLE_DEVICES=0            \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file ./configs/stainer_basic_cmp/exp100.yaml \
    --model_name  model_best_psnr \
    --apply_tta   true            \
    --evaluator   basic
```


## Acknowledgement
Many thanks for these repos for their great contribution!

[https://github.com/quqixun/BCIStainer/tree/main](https://github.com/quqixun/BCIStainer/tree/main)

[https://github.com/JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet)
