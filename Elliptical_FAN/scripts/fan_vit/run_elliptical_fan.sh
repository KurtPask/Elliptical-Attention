### train elliptical tiny code ###
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1298 \
--use_env ../main.py \
--data_dir /../../../../../cm/shared/cuongdc10/datasets/image/ --model elliptical_fan_tiny_12_p16_224 \
-b 128 --sched cosine --epochs 200 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5  \
--remode pixel --reprob 0.25 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 \
--drop-path .1 --img-size 224 --use_wandb \
--output output/elliptical_fan_tiny_12_p16_224/ \
--amp --model-ema --project_name 'FAN' --job_name elliptical_tiny-delta5-maxscale --M-positions 1 2 3 4 5 6 7 8 9 10 11 --show-M \