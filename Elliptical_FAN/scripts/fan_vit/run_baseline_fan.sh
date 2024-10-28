# ### train baseline code ###
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25001 \
--use_env ../main.py \
--data_dir /../../../../../cm/shared/cuongdc10/datasets/image/ --model fan_tiny_12_p16_224 \
-b 128 --sched cosine --epochs 200 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
--remode pixel --reprob 0.25 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
--drop-path .1 --img-size 224 --use_wandb \
--output output/fan_tiny_12_p16_224/ \
--amp --model-ema --project_name 'FAN' --job_name FAN_baseline-resume2 --resume output/fan_tiny_12_p16_224/train/FAN_baseline-wsave-2/checkpoint.pth \