import torch
from torch import nn as nn
from torch.nn.functional import softmax

from ...utils.box_utils import point_form, decode
from .focal import FocalLoss, FocalLossSigmoid
from .smooth_l1 import SmoothL1Loss, AdjustSmoothL1Loss, smooth_l1_loss
from .cross_entropy import cross_entropy


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        FLconf(x, c, f) = (1-P(x, c))**f * Lconf(x, c)
        L(x,c,l,g) = (FLconf(x, c) + αLloc(x,l,g)) / N

        Where:
        - FLconf is Focal Loss as
        - Lconf is the CrossEntropy Loss
        - and Lloc is the SmoothL1 Loss
        - weighted by α which is set to 1 by cross val.

        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, class_activation, negpos_ratio, variance, positive_weight=0, focusing=0,
                 smooth_eps=None, loss_box='SmoothL1', iou_weighting=False):
        """
        parameters
        ----------
        overlap_thresh (float): match as foreground if IOU > overlap_thresh
        overlap_bg_threshold (float): match as background if IOU < overlap_bg_threshold
        negpos_ratio (float): negative:positive ratio for balancing
        variance (float): variance of position and scale of anchor box
        focusing (float): parameter of Focal Loss (0 equals standard cross entropy)
        """
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = negpos_ratio
        self.variance = variance
        self.iou_weighting = iou_weighting
        reduction = 'none' if self.iou_weighting else 'sum'

        if positive_weight > 0:
            # as alpha in focal loss paper
            weight = torch.ones(num_classes) * positive_weight
        else:
            weight = None
        if class_activation == 'sigmoid':
            self.conf_criterion = FocalLossSigmoid(focusing, weight=weight, reduction=reduction, smooth_eps=smooth_eps)
        elif class_activation == 'softmax':
            weight[0] = 1 - positive_weight
            self.conf_criterion = FocalLoss(focusing, weight=weight, reduction=reduction, smooth_eps=smooth_eps)
        else:
            raise NotImplementedError(class_activation)
        self.loss_box = loss_box
        if loss_box == 'SmoothL1':
            self.loss_box = SmoothL1Loss(reduction=reduction)
        elif loss_box == 'AdjustSmoothL1':
            self.loss_box = AdjustSmoothL1Loss(4, reduction=reduction)

    def forward(self, defaults, loc_data, conf_data, priors_label, priors_loc, priors_pos, weights_iou):
        """
        :param defaults: [L, 4]
        :param loc_data: [N, 4, L]
        :param conf_data: [N, C, L]
        :param priors_label: [N, L]
        :param priors_loc: list of [Bi, 4] X N
        :param priors_pos: [N, L]
        :param weights_iou: [N, L]
        :return:
        """
        priors_loc = torch.cat(priors_loc).to(device=loc_data.device, non_blocking=True)
        # in case use_fp16:
        # loc_data = loc_data.float()
        # conf_data = conf_data.float()

        batch_size = loc_data.size(0)
        num_points_per_box = defaults.size(-1)
        num_classes = conf_data.size(1)

        batch_conf = conf_data.transpose(1, 2).contiguous().view(-1, num_classes)  # [N, C, L] --> [N * L, C]

        if self.negpos_ratio > 0:
            # Hard Negative Mining
            batch_loss_c = cross_entropy(batch_conf, priors_label.view(-1), reduction='none')
            loss_c = batch_loss_c.view(batch_size, -1)
            loss_c[priors_label != 0] = 0  # filter out no-background boxes for now
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = priors_pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=priors_pos.size(1) - 1, min=1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = priors_pos.unsqueeze(1).expand_as(conf_data)
            neg_idx = neg.unsqueeze(1).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
            priors_selected = (priors_pos + neg).gt(0)
            targets = priors_label[priors_selected]
            if self.iou_weighting:
                weights_iou = weights_iou[priors_selected]
        else:
            conf_p = batch_conf
            targets = priors_label.view(-1)

        loss_c = self.conf_criterion(conf_p, targets)
        if self.iou_weighting:
            masked_indices = targets.ge(0)
            weights_iou_c = weights_iou.view(-1)[masked_indices]
            loss_c = (loss_c * weights_iou_c).sum()

        # Localization Loss
        loc_data = loc_data.transpose(1, 2)  # [N, 4, L] --> [N, L, 4]
        pos_idx = priors_pos.unsqueeze(-1).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, num_points_per_box)
        if self.loss_box == 'FocalL1':
            # scale the SmoothL1 by predicted probability
            conf_p = conf_data[pos_idx].view(-1, num_classes)
            prob_p = softmax(conf_p, dim=-1)
            prob_t = torch.index_select(prob_p, 0, priors_label[priors_pos].view(-1))
            loss_l = smooth_l1_loss(loc_p, priors_loc, reduction='none') * prob_t
            loss_l = loss_l.sum(dim=-1)
        elif self.loss_box in ('IoU', 'GIoU'):
            defaults_expanded = defaults.unsqueeze(0).expand_as(loc_data)
            defaults_p = defaults_expanded[pos_idx].view(-1, num_points_per_box)
            box_p = point_form(decode(loc_p, defaults_p, self.variance))
            box_t = point_form(decode(priors_loc, defaults_p, self.variance))
            generalized = self.loss_box == 'GIoU'
            loss_l = iou_loss(box_p, box_t, size_average=False, generalized=generalized)
        else:
            loss_l = self.loss_box(loc_p, priors_loc)
            if self.iou_weighting:
                weights_iou_l = weights_iou[priors_pos]
                loss_l = (loss_l.sum(-1) * weights_iou_l).sum()

        # normalize by the number of anchors assigned to a ground-truth box
        num_pos = len(loc_p)
        if num_pos > 0:
            loss_l /= num_pos
            loss_c /= num_pos

        return loss_l, loss_c


def iou_loss(input, target, size_average=True, reduce=True, generalized=False):
    '''
    input: [:, 4] bbox in points form
    target: [:, 4] bbox in points form
    '''
    assert not target.requires_grad
    eps = 1e-8
    area_input = ((input[:, 2] - input[:, 0]) * (input[:, 3] - input[:, 1]))
    area_target = ((target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]))

    max_xy = torch.min(input[:, 2:], target[:, 2:])
    min_xy = torch.max(input[:, :2], target[:, :2])
    wh = torch.clamp((max_xy - min_xy), min=eps)
    inter = wh[:, 0] * wh[:, 1]

    union = area_input + area_target - inter
    iou = inter / union

    if generalized:
        max_xy = torch.max(input[:, 2:], target[:, 2:])
        min_xy = torch.min(input[:, :2], target[:, :2])
        wh = torch.clamp((max_xy - min_xy), min=eps)
        c = wh[:, 0] * wh[:, 1]
        iou = iou - (c - union) / c

    loss = 1 - iou

    if not reduce:
        return loss
    if not size_average:
        return loss.sum()
    return loss.mean()