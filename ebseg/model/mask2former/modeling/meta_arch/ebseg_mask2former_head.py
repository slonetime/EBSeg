import torch
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class EBSeg_Mask2former_Head(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        clip_name: str,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        
        if clip_name == "ViT-B/16":
            self.clip_fusion_feature_layer_idx = [3, 6, 9]
            clip_in_channels = 768
        elif clip_name == "ViT-L-14-336":
            self.clip_fusion_feature_layer_idx = [6, 12, 18]
            clip_in_channels = 1024
            
        self.clip_fusion_layers = nn.ModuleList([nn.Sequential(LayerNorm(clip_in_channels), Conv2d(clip_in_channels, self.pixel_decoder.transformer_in_channels[i], kernel_size=1)) for i in range(3)])

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "clip_name": cfg.MODEL.EBSEG.CLIP_MODEL_NAME
        }
        
    def feature_fusion(self, x, clip_image_features):
        res_features = ['res3', 'res4', 'res5']
        for i in range(len(self.clip_fusion_feature_layer_idx)):
            b,c,h,w = x[res_features[i]].shape
            clip_feature = clip_image_features[self.clip_fusion_feature_layer_idx[i]]
            clip_feature = self.clip_fusion_layers[i](clip_feature)
            clip_feature = F.interpolate(clip_feature, size=(h,w),mode='bilinear',align_corners=False)
            x[res_features[i]] = x[res_features[i]] + clip_feature

        return x

    def forward(self, features, clip_image_features=None, mask=None):
        features = self.feature_fusion(features, clip_image_features)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            raise NotImplementedError
        return predictions
