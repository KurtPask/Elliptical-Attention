# dataset settings
dataset_type = "ADE20KDataset"
data_root = "ADE_data/ADEChallengeData2016"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=(512 * max_ratio, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/training",
        ann_dir="annotations/training",
        pipeline=train_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["images/training", "images/validation"],
        ann_dir=["annotations/training", "annotations/validation"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_clean",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_brightness=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_brightness/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_contrast=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_contrast/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_defocus_blur=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_defocus_blur/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_elastic_transform=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_elastic_transform/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_fog=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_fog/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_frost=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_frost/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_gaussian_blur=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_gaussian_blur/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_gaussian_noise=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_gaussian_noise/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_glass_blur=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_glass_blur/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_impulse_noise=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_impulse_noise/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_jpeg_compression=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_jpeg_compression/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_motion_blur=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_motion_blur/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_pixelate=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_pixelate/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_saturate=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_saturate/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_shot_noise=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_shot_noise/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_snow=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_snow/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_spatter=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_spatter/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_speckle_noise=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_speckle_noise/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    val_zoom_blur=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation_zoom_blur/1",
        ann_dir="annotations/validation",
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="testing",
        pipeline=test_pipeline,
    ),
)
