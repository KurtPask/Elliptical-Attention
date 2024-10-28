export DATASET=ADE_data

##### BASELINE
python -m train --log-dir checkpoints/deit_baseline --dataset ade20k \
--backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name deit-baseline --attn_type 'softmax' \
