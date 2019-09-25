from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .squeezenet import SqueezeNet
from .efficient_net import EfficientNet

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'SqueezeNet', 'EfficientNet']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
