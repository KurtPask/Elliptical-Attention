export DATASET=ADE_data

##### DeiT-Elliptical
python -m train --log-dir checkpoints/deit_elliptical-delta5-2 --dataset ade20k \
--backbone deit_tiny_patch16_224 --decoder mask_transformer --model_name deit-elliptical --attn_type 'elliptical' \
