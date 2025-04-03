#!/usr/bin/env bash

data_dir=/share/data/rust_data/tc_400_75
gctx_data=$data_dir/gctx_dataset

if (($1 == 224)); then
  CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --is_savenii \
    --num_classes=6 \
    --volume_path=$gctx_data \
    --list_dir=$data_dir/lists \
    --output_dir=$data_dir/model_out_224 \
    --max_epochs=150 \
    --img_size=224
elif (($1 == 512)); then
  CUDA_VISIBLE_DEVICES=0 python -W ignore test.py --is_savenii \
    --num_classes=6 \
    --root_path=$gctx_data \
    --volume_path=$gctx_data \
    --list_dir=$data_dir/lists \
    --output_dir=$data_dir/model_out_512 \
    --img_size=512 \
    --batch_size=8 \
    --opts MODEL.PRETRAIN_CKPT './pretrained_ckpt/gcvit_21k_large_512.pth.tar' \
    MODEL.DROP_PATH_RATE 0.1 \
    MODEL.GCVIT.EMBED_DIM 192 \
    MODEL.GCVIT.LAYER_SCALE 1e-5 \
    MODEL.GCVIT.NUM_HEADS "[6, 12, 24, 48]" \
    MODEL.GCVIT.WINDOW_SIZE "[16, 16, 32, 16]" \
    MODEL.GCVIT.MLP_RATIO 2.
else

  echo "Unsupported size $1"
fi
