import os.path as osp

from mmdet.datasets import CocoDataset


class CocoDataset_new(CocoDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

        results['img_fields'] = []

        filenames = []
        for i in range(10):
            filenames.append(results['img_info']['filename'])
        results['ref_imgs_info'] = dict(filenames=filenames)

        # _ = results.pop('img_info')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMultiImagesFromMultiFiles', key_prefix='ref'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'ref_imgs', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ref_filenames', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg')),
]
ann = osp.join(
    osp.dirname(__file__), './data/coco/annotations/instances_val2017.json')

coco = CocoDataset_new(
    ann,
    train_pipeline,
    img_prefix=osp.join(osp.dirname(__file__), './data/coco/val2017/'))
