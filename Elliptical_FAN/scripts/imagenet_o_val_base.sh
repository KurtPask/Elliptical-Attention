CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10017 --nproc_per_node=1 --use_env /root/FAN/eval_OOD.py \
--model fan_tiny_12_p16_224 --data-path /root/data/imagenet/ --output_dir /root/checkpoints/ \
--resume /root/checkpoints/fan_tiny_baseline/train/20240323-204844-fan_tiny_12_p16_224-224/model_best.pth.tar
# --resume /root/checkpoints/fan_tiny_baseline/train/20240323-204844-fan_tiny_12_p16_224-224/model_best.pth.tar
# --resume /root/checkpoints/fan_tiny_robust/train/20240324-173056-fan_tiny_12_p16_224-224/model_best.pth.tar \


