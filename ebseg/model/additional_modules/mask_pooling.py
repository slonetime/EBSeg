import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_pooling(mask, embedding):

    mask_for_pooling = mask
    mask_for_pooling = mask_for_pooling.detach()
    mask_for_pooling = mask_for_pooling.sigmoid()
    mask_for_pooling = mask_for_pooling > 0.5
    denorm = mask_for_pooling.sum(dim=(-1, -2), keepdim=True) + 1e-8
    mask_pooling_embedding = torch.einsum("bnhw,bchw->bnc", mask_for_pooling / denorm, embedding)

    return mask_pooling_embedding


        
