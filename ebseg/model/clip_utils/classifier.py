from typing import List
from torch.nn import functional as F
import torch
from detectron2.utils.registry import Registry
from open_clip.model import CLIP
from torch import nn
from .utils import get_labelset_from_dataset
from open_clip import tokenizer


class PredefinedOvClassifier(nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        templates: List[str] = ["a photo of {}"],
    ):
        super().__init__()
        for name, child in clip_model.named_children():
            if "visual" not in name:
                self.add_module(name, child)
        for name, param in clip_model.named_parameters(recurse=False):
            self.register_parameter(name, param)
        for name, buffer in clip_model.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        self.templates = templates
        self._freeze()

        self.cache_feature = True
        self.cache = {}
        self.num_class_names_cache =  {}
        self.dataset = ' '

    def forward_embeddings(self, category_names: List[str],normalize=True):
        text_embed_bucket = []
        for template in self.templates:
            noun_tokens = tokenizer.tokenize(
                [template.format(noun) for noun in category_names]
            )
            text_inputs = noun_tokens.to(self.text_projection.data.device)
            text_embed = self.encode_text(text_inputs, normalize) 
            text_embed_bucket.append(text_embed)
        
        text_embed = torch.stack(text_embed_bucket).mean(dim=0)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed

    @torch.no_grad()
    def encode_text(self, text, normalize: bool = False):
        with torch.no_grad():
            cast_dtype = self.transformer.get_cast_dtype()

            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.to(cast_dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection #k,C
            return F.normalize(x, dim=-1) if normalize else x

    def get_classifier_by_dataset_name(self, dataset_name: str):
        
        if self.cache_feature:
            if dataset_name != self.dataset:
                self.dataset = dataset_name
                
            if dataset_name not in self.cache:
                category_names = get_labelset_from_dataset(dataset_name)
                cat_embeddings = self.forward_embeddings(category_names)
                self.cache[dataset_name] = cat_embeddings
                        
            cat_embeddings = self.cache[dataset_name]

        return cat_embeddings

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(False)


class LearnableBgOvClassifier(PredefinedOvClassifier):
    def __init__(
        self,
        clip_model: CLIP,
        templates: List[str] = ["a photo of {}"],
    ):
        super().__init__(clip_model, templates)
        self.bg_embed = nn.Parameter(torch.randn(1, self.text_projection.shape[0]))
        nn.init.normal_(
            self.bg_embed,
            std=self.bg_embed.shape[1] ** -0.5,
        )
        self.coco_text_embedding_scoremap = None
        self.coco_text_embedding_distance = nn.PairwiseDistance(p=2)

    def get_classifier_by_dataset_name(self, dataset_name: str):
        
        cat_embedding = super().get_classifier_by_dataset_name(dataset_name)
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)
        cat_embedding_normed = F.normalize(cat_embedding, p=2, dim=-1)
        if self.coco_text_embedding_scoremap == None:
            self.coco_text_embedding_scoremap = torch.einsum("kc,mc->km", cat_embedding_normed[:-1], cat_embedding_normed[:-1]).detach()
            
        return cat_embedding_normed

    def forward(self, dataset_name: str):
        return self.get_classifier_by_dataset_name(dataset_name)