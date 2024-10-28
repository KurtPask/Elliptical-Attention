#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FAN/blob/main/LICENSE

# model_name=$1
# ckpt=$2
# CUDA_VISIBLE_DEVICES=0 ../validate_ood.py \
#   --data ../../ellattack-ly/data/imagenet-r/ \
# 	--model fan_tiny_12_p16_224 \
# 	--img-size 224 \
# 	-b 256 \
# 	-j 32 \
# 	--imagenet_r \
# 	--results-file /aor/checkpoints \
# 	--checkpoint output/fan_tiny_12_p16_224/train/FAN_baseline-resume2/checkpoint.pth \
	#--use-ema 

# CUDA_VISIBLE_DEVICES=0 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-r/ \
# --model elliptical_fan_tiny_12_p16_224 --img-size 224 -b 256 \
# -j 32 --imagenet_r --results-file /aor/checkpoints \
# --checkpoint output/elliptical_fan_tiny_12_p16_224/train/elliptical_FAN-restart2/checkpoint.pth \

### DEIT ###
# CUDA_VISIBLE_DEVICES=6 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-a/imagenet-a/ \
# --model deit_tiny_patch16_224 \
# --img-size 224 \
# -b 128 \
# -j 32 \
# --imagenet_a \
# --results-file /aor/checkpoints/ \
# --checkpoint ../../imagenet/path/output/checkpoint.pth \

### DEIT - ELLIPTICAL ###
CUDA_VISIBLE_DEVICES=6 ../validate_ood.py --data ../../ellattack-ly/data/imagenet-r/ \
--model mahala_deit_overlayers_tiny_patch16_224 \
--img-size 224 \
-b 128 \
-j 32 \
--imagenet_r \
--results-file /aor/checkpoints/ \
--checkpoint ../../imagenet/path/output-M1234567891011-dist-overlayers2-att5e-1/checkpoint.pth \