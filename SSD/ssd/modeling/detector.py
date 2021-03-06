from torch import nn
from ssd.modeling.backbone.vgg import VGG
from ssd.modeling.backbone.basic import BasicModel, Second_Improved_BasicModel
from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd import torch_utils
import torchvision.models as models

from ssd.modeling.backbone.ssd512 import SSD512
from ssd.modeling.backbone.ssd300 import SSD300

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "basic":
        model = Second_Improved_BasicModel(cfg)
        return model
    if backbone_name == "SSD512":
        return SSD512(cfg)
    if backbone_name == "SSD300":
        return SSD300(cfg)
    if backbone_name == "SSD512":
        return SSD512(cfg)
    if backbone_name == "vgg":
        model = VGG(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED:
            print("using pretrained")
            state_dict = load_state_dict_from_url(
                "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth")
            model.init_from_pretrain(state_dict)
        return model
