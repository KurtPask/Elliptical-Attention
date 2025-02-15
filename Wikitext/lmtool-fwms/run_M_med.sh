CUDA_VISIBLE_DEVICES=4,5,6,7 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 \
--adaptive --n_layer 16 --d_model 256 --n_head 8 --d_head 32 --d_inner 2048 --dropout 0.1 \
--dropatt 0.0 --optim adam --lr 0.00025 --max_step 400000 --attn_type 2 --warmup_step 2000 \
--tgt_len 384 --mem_len 0 --eval_tgt_len 384 --batch_size 56 --multi_gpu --M-positions 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --use_wandb \
--project_name 'fourierformer' --seed 1111 --job_name Mfull_med_overlayers-seed-1111-delta5-meanscale --work_dir ./saved_models/Mfull_med_overlayers-delta5-meanscale \
--over-layers --attenuation 5e-2 --show-M --over-layers