checkpoint_config = dict(interval=5)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
#resume_from = '/home/xiaozeqi/cylinder3d_mmdetection3d/epoch_40.pth'
#resume_from = '/home/xiaozeqi/cylinder3d_mmdetection3d/work_dirs/knet3d/epoch_1.pth'
#resume_from = 'sem_pretrain_revisekey.pth'
#resume_from = '/home/xiaozeqi/cylinder3d_mmdetection3d/sem_pretrain_revisekey.pth'
workflow = [('train', 1)]
#workflow = [('val', 1)]
