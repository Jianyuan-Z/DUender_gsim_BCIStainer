# DUender_gsim_BCIStainer
DUender_gsim_BCIStainer
# Train
CUDA_VISIBLE_DEVICES=0          \
python train.py                 \
    --train_dir   ./data/train  \
    --val_dir     ./data/val    \
    --exp_root    ./experiments \
    --config_file ./configs/stainer_basic_cmp/exp100.yaml \
    --trainer     basic
# Evaluate
CUDA_VISIBLE_DEVICES=0            \
python evaluate.py                \
    --data_dir    ./data/test     \
    --exp_root    ./experiments   \
    --output_root ./evaluations   \
    --config_file ./configs/stainer_basic_cmp/exp3.yaml \
    --model_name  model_best_psnr \
    --apply_tta   true            \
    --evaluator   basic
