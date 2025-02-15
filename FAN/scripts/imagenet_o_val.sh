CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10019 --nproc_per_node=1 --use_env ../eval_OOD.py \
--model fan_tiny_12_p16_224 --data-path ../../ellattack-ly/data/imagenet-a/imagenet-o/ --output_dir /aor/checkpoints/ \
--resume output/fan_tiny_12_p16_224/train/FAN_baseline-resume2/checkpoint.pth \
# --resume /root/checkpoints/fan_tiny_baseline/train/20240323-204844-fan_tiny_12_p16_224-224/model_best.pth.tar


