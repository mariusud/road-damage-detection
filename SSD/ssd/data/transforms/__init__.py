from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            #AddRandomPixelValue(), #aug from RoadDamage paper
            #RandomScalePixelValue(), #aug from RoadDamage paper
            ClassSpesificTransforms2(), #aug from RoadDamage paper
            RandomSampleCrop(), #aug from SSD paper
            RandomMirror(), #aug from SSD paper
            ConvertFromInts(),
            ToPercentCoords(),
            #AverageNeighborBlur(), #aug from RoadDamage paper
            #GaussianBlur(), #aug from RoadDamage paper
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
