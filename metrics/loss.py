#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn


def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss


def calc_l1_loss(pred_mid, pred, target, mask=None):
    """ L1 loss wrapper """
    # import pdb; pdb.set_trace()
    mask = mask.unsqueeze(-1)
    loss_mid = F.l1_loss(pred_mid, target, reduction='none')
    loss_mid = (loss_mid * mask).sum() / mask.sum()

    loss_after = F.l1_loss(pred, target, reduction='none')
    loss_after = (loss_after * mask).sum() / mask.sum()
    return loss_mid, loss_after, loss_mid + loss_after