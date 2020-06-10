import torch

from .multi_box import MultiBoxLoss
from .mask import MaskLoss, ElevationLoss


class SSDLoss(torch.nn.Module):
    def __init__(self, priors, variance, num_classes, class_activation, args):
        super().__init__()
        self.priors = priors
        self.loss_box_weight = args.loss_box_weight
        self.loss_guided = args.loss_guided
        self.criterion = MultiBoxLoss(num_classes,
                                      class_activation,
                                      args.loss_negpos_ratio,
                                      variance,
                                      positive_weight=args.loss_positive_weight,
                                      focusing=args.loss_focusing,
                                      smooth_eps=args.label_smoothing,
                                      loss_box=args.loss_box,
                                      iou_weighting=args.loss_iou_weighting)

        self.mask_criterion = MaskLoss(args.loss_mask)
        self.elev_criterion = ElevationLoss()

    def forward(self, outputs, targets):
        masks = targets.get('masks', None)
        elevation = targets.get('elevation', None)
        priors_label = targets.get('priors_label', None)
        priors_loc = targets.get('priors_loc', None)
        priors_pos = targets.get('priors_pos', None)
        weights_iou = targets.get('weights_iou', None)
        loc, conf, mask, ele = outputs
        self.priors = self.priors.to(loc, non_blocking=True)
        loss_l, loss_c = self.criterion(self.priors, loc, conf, priors_label, priors_loc, priors_pos, weights_iou)
        if self.loss_guided:
            r = loss_l.item() / loss_c.item()
            loss_c = loss_c * r

        loss_m = self.mask_criterion(mask, masks)
        loss_e = self.elev_criterion(ele, elevation)
        loss_l = loss_l * self.loss_box_weight

        losses = {'Loc': loss_l,
                  'Conf': loss_c}

        if loss_m is not None:
            losses['Mask'] = loss_m
        if loss_e is not None:
            losses['Elev'] = loss_e

        return losses