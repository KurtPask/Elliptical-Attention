CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10010 \
--use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /../../../cm/shared/cuongdc10/datasets/image/  \
--output_dir path/output/deit-baseline2-memorytest --project_name 'fourierimagenet' --job_name imagenet/deit-tiny-seed0-2 --seed 0 \