from typing import List
import os
import torch
from torch import nn
from torch.nn import functional as F
import open_clip
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head, ShapeSpec
from detectron2.utils.logger import log_first_n
import logging

from .clip_utils import (
    CLIP_surgery_FeatureExtractor,
    CLIP_surgery_RecWithAttnbiasHead,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    get_predefined_templates,
)
from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .additional_modules import mask_pooling
from .segment_anything import sam_model_registry

@META_ARCH_REGISTRY.register()
class EBSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_model: str,
        ssc_loss=None,
        sem_seg_head: nn.Module,
        clip_visual_extractor: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        criterion: SetCriterion,
        size_divisibility: int,
        clip_rec_head: nn.Module = None,
        backbone: nn.Module = None,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        sem_seg_postprocess_before_inference: bool = True,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.criterion = criterion

        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head
        self.ov_classifier = ov_classifier

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        sam_pixel_mean = [123.675, 116.28, 103.53]
        sam_pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("sam_pixel_mean", torch.Tensor(sam_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("sam_pixel_std", torch.Tensor(sam_pixel_std).view(-1, 1, 1), False)

        dataset_names = ['coco_2017_train_stuff_sem_seg', 'coco_2017_test_stuff_sem_seg', 'voc_sem_seg_val','pcontext_sem_seg_val','pcontext_full_sem_seg_val','ade20k_sem_seg_val','ade20k_full_sem_seg_val']
        self.dataset_overlap_maps = {}
        train_class_names = []
        current_path = os.path.abspath(__file__)
        data_class_name_dir = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data/datasets/class_names')
        for dataset_name in dataset_names:
            class_name_file_path = os.path.join(data_class_name_dir, dataset_name + '.txt')
            file = open(class_name_file_path, mode='r')
            class_names = []
            for line in file.readlines():
                per_class_names = line.split(':')[1].split('\n')[0].split(',')
                if dataset_name == 'coco_2017_train_stuff_sem_seg':
                    train_class_names += per_class_names
                else:
                    class_names.append(per_class_names)
                    
            if dataset_name != 'coco_2017_train_stuff_sem_seg':
                category_overlapping_list = []
                for per_class_names in class_names:
                    is_overlapping = not set(train_class_names).isdisjoint(set(per_class_names))
                    category_overlapping_list.append(is_overlapping)
                category_overlapping_map = torch.tensor(category_overlapping_list, dtype=torch.long)
                self.dataset_overlap_maps[dataset_name] = category_overlapping_map
        
        if clip_model == "ViT-B/16":
            self.inference_alpha, self.inference_beta, self.inference_gamma = 0.09, 0.3, 0.7
        elif clip_model == "ViT-L-14-336":
            self.inference_alpha, self.inference_beta, self.inference_gamma = 0.12, 0.4, 0.6
        else:
            raise NotImplementedError
        
        self.ssc_loss = ssc_loss
        if self.ssc_loss == 'mse_loss':
            self.ssc_mse_loss = nn.MSELoss()        
        
    @classmethod
    def from_config(cls, cfg):
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        class_weight = cfg.MODEL.EBSEG.CLASS_WEIGHT
        dice_weight = cfg.MODEL.EBSEG.DICE_WEIGHT
        mask_weight = cfg.MODEL.EBSEG.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight * 0.75,
            "1_loss_ce":class_weight * 0.75,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        weight_dict.update(aux_weight_dict)
        
        ssc_weight = 10
        for i in range(2 * cfg.MODEL.MASK_FORMER.DEC_LAYERS):
            weight_dict.update({f"mse_ssc_loss_{i+1}": ssc_weight})
            
        losses = ["labels", "masks"]

        criterion = SetCriterion(
            num_classes=cfg.MODEL.EBSEG.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.EBSEG.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.EBSEG.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.EBSEG.IMPORTANCE_SAMPLE_RATIO,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.EBSEG.CLIP_MODEL_NAME,
            pretrained=cfg.MODEL.EBSEG.CLIP_PRETRAINED_NAME,
        )
        pixel_mean, pixel_std = (
            preprocess.transforms[-1].mean,
            preprocess.transforms[-1].std,
        )
        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]
        
        output_channels = [128, 256, 512, 1024]
        output_strides = [4, 8, 16, 32]
        output_shape = {name: ShapeSpec(channels=output_channels[i], stride=output_strides[i])
            for i, name in enumerate(["res2", "res3", "res4", "res5"])}
        backbone = sam_model_registry[cfg.MODEL.SAM.MODEL_NAME]()
        output_shape = backbone.output_shape()
        
        first_layer_idx = {"ViT-B/16": 9, "ViT-L-14-336": 18,}
        last_layer_idx = {"ViT-B/16": 9, "ViT-L-14-336": 18,}
        output_mode = 'clip_surgery_san'
        clip_rec_head = CLIP_surgery_RecWithAttnbiasHead(model.visual,
                            first_layer_idx=first_layer_idx[cfg.MODEL.EBSEG.CLIP_MODEL_NAME],
                            frozen_exclude=cfg.MODEL.SAN_HEAD.CLIP_DEEPER_FROZEN_EXCLUDE,
                            sos_token_format=cfg.MODEL.SAN_HEAD.SOS_TOKEN_FORMAT,
                            sos_token_num=cfg.MODEL.SAN_HEAD.NUM_QUERIES,
                            downsample_method=cfg.MODEL.SAN_HEAD.REC_DOWNSAMPLE_METHOD,)
        clip_visual_extractor = CLIP_surgery_FeatureExtractor(model.visual,last_layer_idx=last_layer_idx[cfg.MODEL.EBSEG.CLIP_MODEL_NAME],frozen_exclude=cfg.MODEL.EBSEG.CLIP_FROZEN_EXCLUDE,output_mode=output_mode)

        sem_seg_head = build_sem_seg_head(cfg, output_shape)
        
        ov_classifier = LearnableBgOvClassifier(model, templates=get_predefined_templates(cfg.MODEL.EBSEG.CLIP_TEMPLATE_SET))

        return {
            "clip_model": cfg.MODEL.EBSEG.CLIP_MODEL_NAME,
            "ssc_loss": cfg.MODEL.MODIFIED.SSC_LOSS,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "clip_visual_extractor": clip_visual_extractor,
            "clip_rec_head": clip_rec_head,
            "ov_classifier": ov_classifier,
            "criterion": criterion,
            "size_divisibility": cfg.MODEL.EBSEG.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.EBSEG.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.EBSEG.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.EBSEG.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        }

    def forward(self, batched_inputs):
        inference = not self.training
        
        images_input = [x["image"].to(self.device) for x in batched_inputs]
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images_input]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        images_sam = [(x - self.sam_pixel_mean) / self.sam_pixel_std for x in images_input]
        images_sam = ImageList.from_tensors(images_sam, self.size_divisibility)
        
        clip_input = images.tensor
        clip_input = F.interpolate(clip_input, scale_factor=self.clip_resolution, mode='bilinear')

        image_features_clip = self.clip_visual_extractor(clip_input, inference=inference)
        image_features = self.backbone(images_sam.tensor)
        
        dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
        ov_classifier_weight = self.ov_classifier(dataset_names[0])
        ov_classifier_weight = self.ov_classifier.logit_scale.exp() * ov_classifier_weight 
        
        outputs_head = self.sem_seg_head(image_features, clip_image_features=image_features_clip)
        mask_preds = outputs_head["pred_masks"]
        
        mask_embeds_clip = []
        for attn_bias in outputs_head['attn_biases']:
            mask_embed_clip, clip_final_feature = self.clip_rec_head(features=image_features_clip, attn_bias=attn_bias, inference=inference)
            mask_embeds_clip.append(mask_embed_clip)
        
        mask_logits = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in outputs_head['mask_embeds']
        ]
        mask_logits_clip = [
            torch.einsum("bqc,nc->bqn", mask_emb, ov_classifier_weight)
            for mask_emb in mask_embeds_clip
        ]
        
        if self.training:
            losses = {}
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets, labels = self.prepare_targets(gt_instances, images)
            else:
                targets = None
    
            outputs = {"pred_logits": mask_logits[-1],"pred_masks": mask_preds[-1],"pred_logits_1": mask_logits_clip[-1],"aux_outputs": [{"pred_logits": aux_pred_logits,
                        "pred_masks": aux_pred_masks,"pred_logits_1": aux_mask_logits_clip,}for aux_pred_logits, aux_pred_masks, aux_mask_logits_clip in zip(
                        mask_logits[:-1], mask_preds[:-1], mask_logits_clip[:-1])],}
            
            losses, indices = self.criterion(outputs, targets)
            
            if self.ssc_loss == 'mse_loss':
                ssc_losses = self.get_ssc_loss(labels=labels, indices=indices,
                                               mask_embeds=outputs_head["mask_embeds"], mask_embeds_clip=mask_embeds_clip, 
                                               coco_text_embedding_scoremap=self.ov_classifier.coco_text_embedding_scoremap)
                losses.update(ssc_losses)
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        
        else:
            mask_pred = mask_preds[-1]
            mask_logit = mask_logits[-1]
            mask_logit_clip = mask_logits_clip[-1]
            
            mask_for_pooling = F.interpolate(mask_pred, size=clip_final_feature.shape[-2:],mode='bilinear')
            clip_feature = mask_pooling(mask=mask_for_pooling, embedding=clip_final_feature)
            pooled_logits = torch.einsum('bqc,nc->bqn',clip_feature, ov_classifier_weight)
            
            pooled_logits = pooled_logits[..., :-1].softmax(-1)
            mask_logit = mask_logit[..., :-1].softmax(-1)
            mask_logit_clip = mask_logit_clip[..., :-1].softmax(-1)
            
            dataset_overlap_map = self.dataset_overlap_maps[dataset_names[0]].to(self.device)
            pred_logits_training = (torch.log(1e-8 + (mask_logit ** (1 - self.inference_alpha - self.inference_gamma) * pooled_logits**self.inference_alpha) * mask_logit_clip**self.inference_gamma)
                    * dataset_overlap_map)
            pred_logits_new = (torch.log(1e-8 + (mask_logit ** (1 - self.inference_beta - self.inference_gamma) * pooled_logits**self.inference_beta * mask_logit_clip**self.inference_gamma))
                    * (1 - dataset_overlap_map))
            mask_cls_results = pred_logits_training + pred_logits_new

            binary_probs1 = F.softmax(mask_logits[-1], dim=-1)[..., -1:]
            binary_probs0 = 1 - binary_probs1
            mask_cls_results = torch.cat([F.softmax(mask_cls_results, dim=-1) * binary_probs0, binary_probs1], dim=-1)
            mask_logit = torch.log(mask_cls_results + 1e-8)
            
            mask_pred = F.interpolate(mask_pred,size=(images.tensor.shape[-2], images.tensor.shape[-1]),mode="bilinear",align_corners=False,)
            
            processed_results = []
            for mask_logit_, mask_pred_, input_per_image, image_size in zip(
                mask_logit, mask_pred, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_ = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_, image_size, height, width
                    )
                    mask_logit_ = mask_logit_.to(mask_pred_)
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_logit_, mask_pred_
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        labels = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            labels.append(targets_per_image.gt_classes)
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets, labels

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def get_ssc_loss(self, labels, indices, mask_embeds, mask_embeds_clip, coco_text_embedding_scoremap):
        b,q,_ = mask_embeds[-1].shape
        
        ssc_losses = {}
        i = 0
            
        for mask_embeddings in [mask_embeds]:
            for embeddings, indice in zip(mask_embeddings[-1:], indices[-1:]):
                i += 1
                loss_per_block = []
                for bs in range(b):
                    mask_idx, label_idx = indice[bs]
                    mask_idx = mask_idx.to(self.device).detach_()
                    label_idx = label_idx.to(self.device).detach_()
                    
                    matched_mask_embeddings = torch.index_select(embeddings[bs], dim=0, index=mask_idx)
                    labels_in_order = labels[bs].to(self.device).gather(dim=0, index=label_idx)
                    labels_in_order = labels_in_order.detach_()
                    
                    matched_embeddings_dis = torch.einsum("xc,mc->xm", matched_mask_embeddings, matched_mask_embeddings)
                    matched_text_embeddings_dis = torch.index_select(torch.index_select(coco_text_embedding_scoremap, dim=0, index=labels_in_order), dim=1, index=labels_in_order)
                    
                    loss = self.ssc_mse_loss(matched_embeddings_dis, matched_text_embeddings_dis)

                    loss_per_block.append(loss)
                    
                loss_per_block = sum(loss_per_block) / b
                
                if torch.any(torch.isnan(loss_per_block)):
                    loss_per_block = torch.tensor([0.], requires_grad=True, dtype=torch.float32).to(self.device)
                    log_first_n(logging.INFO, 'detect nan ssc loss, set this batch\'s ssc loss to 0!', n=2000)
                    
                ssc_losses[f"mse_ssc_loss_{i}"] = loss_per_block
                
        return ssc_losses
    
    @property
    def device(self):
        return self.pixel_mean.device