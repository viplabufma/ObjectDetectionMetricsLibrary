model = dict(
    type='CenterNet',
    # use caffe img_norm
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        # There is a chance to get 40.3 after switching init_cfg,
        # otherwise it is about 39.9~40.1
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        hm_min_radius=4,
        hm_min_overlap=0.8,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        soft_weight_on_reg=False,
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
#dataset_type = 'CocoDataset'
data_root = '/home/matheus_levy/workspace/dataset/mdetection_dataset/'
backend_args = None
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('Parasit Eggs', )),
        data_root='/home/matheus_levy/workspace/dataset/mdetection_dataset/',
        ann_file='/home/matheus_levy/workspace/dataset/mdetection_dataset/train/1_class_annotation.coco.json',
        data_prefix=dict(img='/home/matheus_levy/workspace/dataset/mdetection_dataset/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RandomResize',
                scale=[(1111, 480), (1111, 800)],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('Parasit Eggs', )),
        data_root='/home/matheus_levy/workspace/dataset/mdetection_dataset/',
        ann_file='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid/1_class_annotation.coco.json',
        data_prefix=dict(img='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1111, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('Parasit Eggs', )),
       data_root='/home/matheus_levy/workspace/dataset/mdetection_dataset/',
        ann_file='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid/1_class_annotation.coco.json',
        data_prefix=dict(img='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1111, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid/1_class_annotation.coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/matheus_levy/workspace/dataset/mdetection_dataset/valid/1_class_annotation.coco.json',
    metric='bbox',
    format_only=False,
    outfile_prefix='/home/matheus_levy/workspace/RPN_YOLO_Center_Retina/predicts_bbox_Centernet_val',
    backend_args=None,
    )
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[3, 7],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=0.00001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)))

auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],    
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/home/matheus_levy/workspace/models_and_checkpoints/CenterNet/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth'
resume = False
launcher = 'none'
work_dir = '/home/matheus_levy/workspace/RPN_YOLO_Center_Retina/work_dir_CenterNEt'