from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from open_clip.transformer import VisionTransformer
from detectron2.layers import ShapeSpec
from open_clip.transformer import ResidualAttentionBlock
from detectron2.layers import Conv2d
from collections import OrderedDict

def downsample2d(src, target_shape, method="nearest"):
    if method in ["bicubic", "bilinear", "nearest"]:
        src = F.interpolate(src, size=target_shape, mode=method, align_corners=False)
    elif method == "avg":
        src = F.adaptive_avg_pool2d(src, output_size=target_shape)
    elif method == "max":
        src = F.adaptive_max_pool2d(src, output_size=target_shape)
    return src


def resize_pos_embed2d(
    posemb,
    src_shape,
    tgt_shape,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=False,
):
    """interpolate positional embedding from src_shape to tgt_shape. posemb: [N,L,C]"""
    if src_shape == tgt_shape:
        return posemb
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    posemb_grid = posemb_grid.permute(0, 2, 1).reshape(
        1, -1, src_shape[0], src_shape[1]
    )

    posemb_grid = F.interpolate(
        posemb_grid,
        size=tgt_shape,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, tgt_shape[0] * tgt_shape[1], -1
    )
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

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


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x, attn_mask=None, inference=True):
        
        B, N, C = x.shape
            
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
                
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask != None:
            attn_ori += attn_mask[0]
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))
        
        if not inference:
            return x_ori
        else:
            # replace k & q by v
            k = v
            q = k
            scale = self.scale
            attn = (q @ k.transpose(-2, -1)) * scale
            if attn_mask != None:
                attn += attn_mask[1]
            attn = (attn).softmax(dim=-1)
            attn = self.attn_drop(attn)
            # clip_surgery
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
            
            return [x, x_ori]

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Surgery_ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask=None, inference=True):
        
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            if inference:
                x, x_ori = self.attn(x, attn_mask=attn_mask, inference=inference)
                return [x.transpose(0, 1), x_ori.transpose(0, 1)]
            else:
                x = self.attn(x, attn_mask=attn_mask, inference=inference)
                return x.transpose(0, 1)
            
    def forward(self, x, attn_mask=None, inference=True):

        if not inference:
            x = x + self.attention(self.ln_1(x), attn_mask=attn_mask, inference=inference)
            x = x + self.mlp(self.ln_2(x))
            return x
        else:
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori), attn_mask=attn_mask, inference=inference)
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x), attn_mask=attn_mask, inference=inference)
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]


class ClipOutput(dict):
    def __init__(self, spacial_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacial_shape = spacial_shape

    def save(self, idx: int, clip_feat: torch.Tensor):
        l, n, c = clip_feat.shape
        self[idx] = (
            clip_feat[1:].permute(1, 2, 0).contiguous().reshape(n, c, *self.spacial_shape).contiguous()
        )
        self[f"{idx}_cls_token"] = clip_feat[0:1]


class CLIP_surgery_FeatureExtractor(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        last_layer_idx: int = -1,
        frozen_exclude=[],
        output_mode='clip_surgery_san',
    ):
        super().__init__()
        self.output_tokens = visual_encoder.output_tokens
        self.image_size = visual_encoder.image_size
        self.patch_size = visual_encoder.patch_size
        self.grid_size = visual_encoder.grid_size
        self.num_features = visual_encoder.ln_pre.normalized_shape[0]

        self.input_patchnorm = visual_encoder.input_patchnorm
        self.patchnorm_pre_ln = visual_encoder.patchnorm_pre_ln
        self.conv1 = visual_encoder.conv1

        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        self.patch_dropout = visual_encoder.patch_dropout
        self.ln_pre = visual_encoder.ln_pre

        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj

        if last_layer_idx == -1:
            self.resblocks = visual_encoder.transformer.resblocks
            self.last_output_idx = len(self.resblocks) + 1
        else:
            self.resblocks = visual_encoder.transformer.resblocks[:last_layer_idx]
            self.last_output_idx = last_layer_idx + 1
            
        # add CLIP surgery resblocks
        n_head = visual_encoder.transformer.resblocks[0].attn.num_heads
        self.surgery_start_idx = int(visual_encoder.transformer.layers / 6)
        for i in range(self.surgery_start_idx, len(self.resblocks)):
            resblock = Surgery_ResidualAttentionBlock(d_model=self.num_features, n_head=n_head)
            attn = Attention(self.num_features, self.num_features, n_head, True)
            attn.qkv.weight.data = self.resblocks[i].attn.in_proj_weight.clone()
            attn.qkv.bias.data = self.resblocks[i].attn.in_proj_bias.clone()
            attn.proj.weight.data = self.resblocks[i].attn.out_proj.weight.clone()
            attn.proj.bias.data = self.resblocks[i].attn.out_proj.bias.clone()
            resblock.attn = attn
            resblock.ln_1 = self.resblocks[i].ln_1
            resblock.mlp = self.resblocks[i].mlp
            resblock.ln_2 = self.resblocks[i].ln_2
            self.resblocks[i] = resblock
        
        self.frozen_exclude = frozen_exclude
        self._freeze(self.frozen_exclude)
        
        self.output_mode = output_mode

    def forward(self, x: torch.Tensor, inference: bool = True):
        if self.input_patchnorm:
            raise NotImplementedError("input_patchnorm is not implemented yet.")
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            n, _, h, w = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1).contiguous()  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        pos_embed = self.positional_embedding.to(x.dtype)
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], self.grid_size, (h, w))[0]
        x = x + pos_embed

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2).contiguous()

        outputs = ClipOutput(spacial_shape=(h, w))
        outputs.save(0, x)
        for i, resblock in enumerate(self.resblocks, start=1):
            if i > self.surgery_start_idx:
                x = resblock(x, inference=inference)
            else:
                x = resblock(x)
            if isinstance(x, list):
                outputs.save(i, x[1])
            else:
                outputs.save(i, x)
                
        if self.output_mode == 'clip_last_feature':
            output = self.ln_post(x[0].permute(1, 0, 2).contiguous())
            output = output @ self.proj
            output = F.normalize(output, dim=-1)
            output = output[:,1:,:].permute(0, 2, 1).contiguous().reshape(n,self.proj.shape[1],h,w)
            
            return output
        elif self.output_mode == 'clip_surgery_san':
            # return CLIP surgery features and the last CLIP ori feature
            if inference:
                outputs['last_surgery'] = x[0]
            return outputs

        else:
            raise NotImplementedError

    def _freeze(self, frozen_exclude):
        if "all" in frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = False

    @property
    def output_shapes(self):
        return {
            i: ShapeSpec(channels=self.num_features)
            for i in range(self.last_output_idx)
        }

    @property
    def size_divisibility(self):
        return self.patch_size[0]


class CLIP_surgery_RecWithAttnbiasHead(nn.Module):
    def __init__(
        self,
        visual_encoder: VisionTransformer,
        first_layer_idx: int = 0,
        frozen_exclude: List[str] = [],
        sos_token_format: str = "cls_token",
        sos_token_num: int = 100,
        downsample_method: str = "bilinear",
    ):
        super().__init__()
        self.output_tokens = visual_encoder.output_tokens
        self.output_dim = visual_encoder.output_dim
        self.first_layer_idx = first_layer_idx
        self.downsample_method = downsample_method

        if first_layer_idx < 0:
            raise NotImplementedError("first_layer_idx < 0 is not implemented yet.")
        self.resblocks = visual_encoder.transformer.resblocks[first_layer_idx:]
        
        d_model = visual_encoder.ln_pre.normalized_shape[0]
        num_head = visual_encoder.transformer.resblocks[0].attn.num_heads
        
        for i in range(visual_encoder.transformer.layers - first_layer_idx):
            resblock = Surgery_ResidualAttentionBlock(d_model=d_model, n_head=num_head)
            attn = Attention(d_model, d_model, num_head, True)
            attn.qkv.weight.data = self.resblocks[i].attn.in_proj_weight.clone()
            attn.qkv.bias.data = self.resblocks[i].attn.in_proj_bias.clone()
            attn.proj.weight.data = self.resblocks[i].attn.out_proj.weight.clone()
            attn.proj.bias.data = self.resblocks[i].attn.out_proj.bias.clone()
            resblock.attn = attn
            resblock.ln_1 = self.resblocks[i].ln_1
            resblock.mlp = self.resblocks[i].mlp
            resblock.ln_2 = self.resblocks[i].ln_2
            self.resblocks[i] = resblock
        
        self.global_average_pool = visual_encoder.global_average_pool
        self.attn_pool = visual_encoder.attn_pool        
        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj

        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude
        
        self.frozen_exclude = ['attn_bias_', ]
        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if any([exclude in name for exclude in frozen_exclude]):
                param.requires_grad = True
        
    def forward(self, features, attn_bias=None, inference=True):
        cls_token = features[f"{self.first_layer_idx}_cls_token"]
        pix_feat = features[self.first_layer_idx]
        n, c, h, w = pix_feat.shape
        x = torch.cat([cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1).contiguous()])
        sos_token = cls_token.repeat(self.sos_token_num, 1, 1)
        x = torch.cat([sos_token, x], dim=0)
        if inference:
            last_surgery_feat = features['last_surgery']
            sos_surgery_token = last_surgery_feat[0:1,:,:].repeat(self.sos_token_num, 1, 1)
            x_surgery = torch.cat([sos_surgery_token, last_surgery_feat], dim=0)
            x = [x_surgery, x]
        
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, attn_mask=attn_biases[i], inference=inference)

        if not inference:
            sos_token = x[: self.sos_token_num]
        else:
            sos_token = x[1][: self.sos_token_num]
        sos_token = sos_token.permute(1, 0, 2).contiguous()
        sos_token = self.ln_post(sos_token)
        if self.proj is not None:
            sos_token = sos_token @ self.proj
        sos_token = F.normalize(sos_token, dim=-1)
        
        if not inference:
            return sos_token, None
        else:
            image_feature = x[0][self.sos_token_num:]
            image_feature = image_feature.permute(1, 0, 2).contiguous()
            image_feature =  self.ln_post(image_feature)
            if self.proj is not None:
                image_feature = image_feature @self.proj
            image_feature = F.normalize(image_feature, dim=-1)
            image_feature = image_feature[:,1:,:].permute(0, 2, 1).contiguous().reshape(n,self.proj.shape[1],h,w)

            return sos_token, image_feature

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        if not isinstance(attn_biases, list):
            attn_biases = [attn_biases]
        for attn_bias in attn_biases:
            n, num_head, num_sos, h, w = attn_bias.shape
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method=self.downsample_method,
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)
            true_num_head = self.resblocks[0].attn.num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
            new_attn_bias[:, :num_sos] = -100
            
            new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
            
            new_attn_bias[:num_sos, num_sos] = -100
            new_attn_bias = (new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone())
            plain_attn_bias = new_attn_bias
            plain_attn_bias1 = plain_attn_bias
            plain_attn_bias1[..., :num_sos, -L:] = -100
            new_attn_bias[..., :num_sos, -L:] = attn_bias
            
            new_attn_bias1 = [new_attn_bias.reshape(n, true_num_head, num_sos + 1 + L, num_sos + 1 + L), plain_attn_bias.reshape(n, true_num_head, num_sos + 1 + L, num_sos + 1 + L), plain_attn_bias1.reshape(n, true_num_head, num_sos + 1 + L, num_sos + 1 + L)]

            formatted_attn_biases.append(new_attn_bias1)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]
            
        return formatted_attn_biases
