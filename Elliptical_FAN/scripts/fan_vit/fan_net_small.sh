#!/user/bin/env bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# set -e
# set -x
# export NCCL_LL_THRESHOLD=0


# rank_num=$1
# rank_index=$2
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=$rank_num \
# 	--node_rank=$rank_index --master_addr="10.192.24.22" --master_port=3349 \
# 	main.py  ../imagenet_raw/ --model fan_small_12_p16_224 -b 32 --sched cosine --epochs 300 \
# 	--opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
# 	--model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel \
# 	--reprob 0.25 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
# 	--drop-path .2 --img-size 224 --mixup 0.8 --cutmix 1.0 \
# 	--smoothing 0.1 \
# 	--output ../fan_small_12_p16_224/ \
# 	--amp --model-ema \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1298 \
--use_env ../main.py \
--data_dir /../../../../../cm/shared/cuongdc10/datasets/image/ --model elliptical_fan_small_12_p16_224 \
-b 128 --sched cosine --epochs 300 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
--remode pixel --reprob 0.25 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
--drop-path .2 --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --grad-median --use_wandb \
--output output/elliptical_fan_small_12_p16_224/ \
--amp --model-ema --project_name 'FAN' --job_name elliptical_small-kv-gradmed --M-positions 1 2 3 4 5 6 7 8 9 10 11 --show-M \