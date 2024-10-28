CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master-port=1839 \
--use_env main.py --model deit_mahala_overlayers_tiny_patch16_224 --batch-size 256 --data-path /../../../cm/shared/cuongdc10/datasets/image/  \
--output_dir path/output-Mfull-tiny-dist-overlayers2-delta5-memorytest --project_name 'fourierimagenet' --job_name M-tiny-seed0-Mfull-dist-overlayers2-delta5-memorytest \
--seed 0 --M-positions 1 2 3 4 5 6 7 8 9 10 11 --show-M --attenuation 5e-1