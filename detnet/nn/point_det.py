"""
Objects as Points
https://github.com/xingyizhou/CenterNet
"""

import torch
from torch import nn
import numpy as np
import math
from torchvision.ops import nms
from .modules.pose_resnet import PoseResNet
from .modules.bounding_box import BoxList


def get_kp_torch(pred,conf,topk=100):
    c, h, w = pred.shape
    pred = pred.view(-1)
    pred[pred < conf] = 0
    topk = min(len(pred),topk)
    score,topk_idx = torch.topk(pred,k=topk)
    cls = (topk_idx / (h*w))
    channel = topk_idx - cls * h * w
    x = channel % w
    y = channel / w
    return x.view(-1),y.view(-1),cls.view(-1)


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


class PointDet(nn.Module):
    def __init__(self, model, classnames, head_conv=256):
        super().__init__()
        self.classnames = classnames
        self.model = model
        self.num_class = len(classnames) -1

        heads = {'hm': self.num_class, 'wh': 2}
        for head, classes in heads.items():
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(model.out_channels, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                #else:
                #    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(model.out_channels, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                #else:
                #    fill_fc_weights(fc)
            setattr(self, head, fc)

    def decoder(self, heatmap, offsets, img_size, conf=0.05, nms_threshold=0.5, topk=200):
        assert len(heatmap.shape) == 3, 'no batch support'
        class_num,h,w = heatmap.shape

        heatmap = heatmap.data.cpu()
        offsets = offsets.data.cpu()

        x_list, y_list, c_list = get_kp_torch(heatmap, conf=conf, topk=topk)
        cls, idx = torch.sort(c_list)

        bboxes = []
        scores = []
        labels = []

        for i in range(class_num):
            mask = idx[cls.eq(i)]

            if len(mask) > 0:
                y = y_list[mask]
                x = x_list[mask]
                c = c_list[mask]
                score = heatmap[c, y, x]
                mask_score = score > conf

                if mask_score.sum() <= 0:
                    continue
                score = score[mask_score]
                x = x[mask_score]
                y = y[mask_score]
                bw = (offsets[0, y, x] + 0.5) * w / 2
                bh = (offsets[1, y, x] + 0.5) * h / 2

                x1 = torch.clamp(x.float() - bw, 0, w).unsqueeze(dim=1)
                y1 = torch.clamp(y.float() - bh, 0, h).unsqueeze(dim=1)
                x2 = torch.clamp(x.float() + bw, 0, w).unsqueeze(dim=1)
                y2 = torch.clamp(y.float() + bh, 0, h).unsqueeze(dim=1)

                bbox = torch.cat((x1, y1, x2, y2), dim=1)

                keep = nms(bbox, score, nms_threshold)

                scores += score[keep].tolist()
                labels += [i for _ in range(len(keep))]
                bboxes += bbox[keep].tolist()

        if len(bboxes) > 0:
            scores = np.asarray(scores)
            bboxes = np.asarray(bboxes)
            labels = np.asarray(labels)

            box = BoxList(bboxes, (w, h))
            box.add_field('scores', scores)
            box.add_field('labels', labels)
        else:
            box = BoxList(np.asarray([[0., 0., 1., 1.]]), (w, h))
            box.add_field('scores', np.asarray([0.]))
            box.add_field('labels', np.asarray([0.]))
        box.resize(img_size)
        #print(img_size,len(bboxes))
        return box

    def forward(self, x):
        x = self.model(x)
        pred_hm = self.hm(x)
        pred_wh = self.wh(x)
        pred_hm = pred_hm.sigmoid()
        return torch.cat((pred_hm, pred_wh), dim=1)

    def criterion(self, args):
        return self.loss

    def loss(self, output, target):
        pred_hm, pred_wh = output[:, :-2], output[:, -2:]
        device = pred_hm.get_device()
        target = target['heatmap']
        target = target.to(device)

        gt_hm = target[:, :self.num_class, :, :]
        gt_hm_max_val, _ = torch.max(gt_hm, dim=1, keepdim=True)
        gt_wh = target[:, self.num_class:, :, :]

        wh_mask = gt_hm_max_val == 1
        num_wh = wh_mask.sum() + 1e-4
        wh_mask = wh_mask.expand_as(pred_wh)
        loc_pred = pred_wh[wh_mask]
        loc_target = gt_wh[wh_mask]

        loss_hm = self._neg_loss(pred_hm, gt_hm)
        loss_wh = nn.functional.l1_loss(loc_pred, loc_target,reduction='sum') / num_wh

        return {'hm': loss_hm, 'wh': loss_wh}

    def _neg_loss(self,pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

    def predict(self, x, conf_thresh=0.05, nms_thresh=0.5, top_k=200, **kwargs):
        with torch.no_grad():

            net_param = next(self.parameters())
            x = x.to(net_param)

            output = self(x)
            pred_hm, pred_wh = output[:, :-2], output[:, -2:]
            #解码之后坐标是相对
            #POOL NMS
            pred_hm = pool_nms(pred_hm)
            output = []
            for bs in range(pred_hm.shape[0]):
                hm = pred_hm[bs]
                off = pred_wh[bs]
                box_list = self.decoder(hm, off, [1, 1], conf_thresh, nms_thresh, top_k)

                # BoxList --> list of array
                scores = box_list.get_field('scores')
                scores = np.asarray(scores)[:, None]
                box = box_list.box
                center = (box[:, 0:2] + box[:, 2:4]) / 2
                wh = (box[:, 2:4] - box[:, 0:2])
                bbox = np.hstack((scores, center, wh))
                labels = box_list.get_field('labels')
                bbox_cls = []
                for cls in range(len(hm)):
                    bbox_cls.append(bbox[labels==cls])
                output.append(bbox_cls)
            return output


def create_model(classnames, basenet, frozen_bn=False):
    model = PoseResNet(basenet=basenet, frozen_bn=frozen_bn)
    return PointDet(model, classnames)


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return gaussian


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,radius - left:radius + right]


    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])

    masked_regmap = (1 - idx) * masked_regmap +  idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap

    return regmap


def get_heatmap(box_list, scale_idx ,scale_list, class_num,radius=1,smooth=1):

    anchors = [(0.5 ,0.5)]

    w,h = box_list.size
    heatmap = np.zeros((class_num ,h, w), dtype=np.float32)
    dense_offset = np.zeros((2, h, w), dtype=np.float32)

    base = radius
    labels = box_list.get_field('labels')

    scale = scale_list[scale_idx]

    scale_range = np.linspace(0,max(h,w)*scale,len(scale_list)+1)

    for gt,k in zip(box_list.box,labels):
        bh =  gt[3] - gt[1]
        bw =  gt[2] - gt[0]

        board = max(bh,bw)*scale

        #TODO support VOC
        if min(bh,bw)*scale < 20 or bh <= 0 or bw <= 0 or max(bh,bw) / min(bh,bw) > 3:
            continue

        if scale_range[scale_idx] < board < scale_range[scale_idx+1]:

            radius = gaussian_radius((math.ceil(bh),math.ceil(bw)))
            radius = max(base, int(radius))

            cc = [max(radius+1, int((gt[2] + gt[0]) / 2)), max(radius+1, int((gt[3] + gt[1]) / 2))]
            cc = [min(w-radius-1, cc[0]), min(h-radius-1, cc[1])]

            #TODO confidence encoding Label smooth
            draw_gaussian(heatmap[k,:,:], cc, radius) * smooth

            #TODO value smooth
            #for i in range(len(anchors)):
            value = float(gt[2] - gt[0]) / w - 0.5,float(gt[3] - gt[1]) / h - 0.5
            draw_dense_reg(dense_offset[0:2, :, :], heatmap.max(axis=0), cc, value, radius)


            #distanceUL = (float(cc[0] - gt[0]) / w - 0.5,float(cc[1] - gt[1]) / h - 0.5)
            #distanceDR = (float(gt[2] - cc[0]) / w - 0.5,float(gt[3] - cc[1]) / h - 0.5)
            #draw_dense_reg(dense_offset[0:2, :, :], heatmap.max(axis=0), cc, distanceUL, radius)
            #draw_dense_reg(dense_offset[2:4, :, :], heatmap.max(axis=0), cc, distanceDR, radius)



    dense_wh_mask = np.concatenate([heatmap,  dense_offset], axis=0)
    return dense_wh_mask
