_base_ = [
    './datasets/coco_ms.py',
    './models/polarnext_res101_torch.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.PolarNeXt.model'], allow_failed_imports=False)

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)

train_dataloader = dict(
    batch_size=4,
    num_workers=4
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

_base_.visualizer.vis_backends = dict(type='TensorboardVisBackend')
