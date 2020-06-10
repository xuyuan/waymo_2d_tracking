# -*- coding: utf-8 -*-
import torch
from torchvision.ops import nms as torchvision_nms
from shapely.geometry import asPolygon
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Returns: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    if boxes.size(-1) == 5:
        x, y = boxes[..., 0], boxes[..., 1]
        w = boxes[..., 2] / 2  # half of width
        h = boxes[..., 3] / 2  # half of height
        a = boxes[..., 4]
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        w_cos = w * cos_a
        w_sin = w * sin_a
        h_cos = h * cos_a
        h_sin = h * sin_a
        points = torch.stack((x-w_cos+h_sin, y-w_sin-h_cos,
                              x-w_cos-h_sin, y-w_sin+h_cos,
                              x+w_cos-h_sin, y+w_sin+h_cos,
                              x+w_cos+h_sin, y+w_sin-h_cos), dim=-1)
        return points

    center = boxes[..., :2]
    half_wh = boxes[..., 2:4] * 0.5
    return torch.cat((center - half_wh,     # xmin, ymin
                      center + half_wh), -1)  # xmax, ymax


def rbox_2_bbox(rbox):
    """
    Parameters
    ----------
    rbox: [..., 8]

    Returns: [..., 4]
    -------

    """
    x = rbox[..., 0::2]
    y = rbox[..., 1::2]
    xmin, _ = torch.min(x, dim=1)
    xmax, _ = torch.max(x, dim=1)
    ymin, _ = torch.min(y, dim=1)
    ymax, _ = torch.max(y, dim=1)
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    xy0 = boxes[:, :2]
    xy1 = boxes[:, 2:4]
    return torch.cat(((xy0 + xy1) * 0.5,  # cx, cy
                      xy1 - xy0),   # w, h
                     dim=1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, points_a, bbox_points_a,
            box_b, points_b, bbox_points_b, area_b=None):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

    if box_a.size(1) == 5:
        return iou_rbox(box_a, points_a, bbox_points_a,
                        box_b, points_b, bbox_points_b, area_b)
    else:
        return iou_bbox(box_a, points_a, box_b, points_b, area_b)


def iou_bbox(box_a, points_a, box_b, points_b, area_b=None):
    inter = intersect(points_a, points_b)

    area_a = (box_a[:, 2] * box_a[:, 3]).unsqueeze(1).expand_as(inter)  # [A,B]
    if area_b is None:
        area_b = box_b[:, 2] * box_b[:, 3]
    area_b = area_b.unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    iou = inter / union  # [A,B]
    return iou


def jaccard_bbox(bbox_a, bbox_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
    bbox_points_a = point_form(bbox_a)
    bbox_points_b = point_form(bbox_b)
    return iou_bbox(bbox_a, bbox_points_a, bbox_b, bbox_points_b)


def iou_rbox(box_a, points_a, bbox_points_a, box_b, points_b, bbox_points_b, area_b=None, top_k=6):
    # compute intersection between bounding boxes first
    iou = iou_bbox(box_a, bbox_points_a, box_b, bbox_points_b, area_b)

    sorted_iou, idx = iou.sort(dim=1)  # sort in ascending order
    iou_threshold = sorted_iou[:, -top_k:-top_k+1]

    valid_inter = (iou - iou_threshold) > 0
    inter_nonzero = torch.nonzero(valid_inter)

    # compute intersection between rotation boxes when there is bbox intersection
    iou.zero_()
    for i, j in inter_nonzero:
        pa = asPolygon(points_a[i].view(-1, 2))
        pb = asPolygon(points_b[j].view(-1, 2))
        intersection = pa.intersection(pb).area
        union = box_a[i, 2] * box_a[i, 3] + box_b[j, 2] * box_b[j, 3] - intersection
        iou[i, j] = intersection / union
    return iou


def match(truths, truths_points, truths_bbox_points, priors, priors_points, priors_bbox_points, priors_area=None):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
    Return:
        best_truth_idx: matched index
        best_truth_overlap: (tensor) best matched overlap.
    """
    # jaccard index
    overlaps = jaccard(
        truths, truths_points, truths_bbox_points,
        priors, priors_points, priors_bbox_points, priors_area
    )
    # (Bipartite Matching)
    # [num_objects] best prior for each ground truth
    #best_prior_overlap, best_prior_idx = overlaps.max(1)
    # [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)

    # ensure every gt matches with its prior of max overlap
    # TODO: this introduce unwanted results: prior is not closes to what it should detect, and instability of max
    #best_truth_idx[best_prior_idx] = torch.arange(len(best_prior_idx), dtype=torch.long, device=best_truth_idx.device)
    #best_truth_overlap[best_prior_idx] = best_prior_overlap
    
    return best_truth_idx, best_truth_overlap


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in offset-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = matched[:, :2] - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = matched[:, 2:4] / priors[:, 2:4]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    loc = [g_cxcy, g_wh]
    if len(variances) == 3:
        # rotation
        g_r = (matched[:, 4:5] - priors[:, 4:5]) / variances[2]
        loc.append(g_r)

    return torch.cat(loc, 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:4]
    wh = priors[:, 2:4] * torch.exp(loc[:, 2:4] * variances[1])
    boxes = [cxcy, wh]
    if len(variances) == 3:
        # rotation
        r = priors[:, 4:5] + loc[:, 4:5] * variances[2]
        boxes.append(r)
    boxes = torch.cat(boxes, 1)
    return boxes


def nms_rboxes(rboxes, scores, overlap, top_k):
    keep = []

    sorted_scores, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals

    boxes = rbox_2_bbox(rboxes[idx])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    while idx.size(0) > 1:
        i = idx[-1].item()  # index of current largest val
        keep.append(i)
        box = boxes[-1]
        box_area = area[-1]

        idx = idx[:-1]  # remove kept element from view
        boxes = boxes[:-1]
        area = area[:-1]
        x1 = torch.clamp(boxes[:, 0], min=box[0].item())
        y1 = torch.clamp(boxes[:, 1], min=box[1].item())
        x2 = torch.clamp(boxes[:, 2], min=box[2].item())
        y2 = torch.clamp(boxes[:, 3], min=box[3].item())
        w = torch.clamp(x2 - x1, min=0.0)
        h = torch.clamp(y2 - y1, min=0.0)
        inter = w * h
        union = (area - inter) + box_area
        IoU = inter / union

        possible_overlaps = torch.nonzero(IoU > overlap)
        if len(possible_overlaps) > 0:
            # load rboxes of next highest vals
            rbox = asPolygon(rboxes[i].view(-1, 2))
            robox_area = rbox.area

            for j in possible_overlaps:
                k = idx[j]
                rbox_other = asPolygon(rboxes[k].view(-1, 2))
                inter = rbox.intersection(rbox_other).area
                if inter > 0:
                    union = robox_area + rbox_other.area - inter
                    IoU[j] = inter / union

        # keep only elements with an IoU <= overlap
        iou_le_overlap = IoU.le(overlap)
        idx = idx[iou_le_overlap]
        boxes = boxes[iou_le_overlap]
        area = area[iou_le_overlap]

    if idx.size(0) > 0:
        keep.append(idx[-1].item())

    scores = scores[keep]

    return keep, scores


# https://github.com/fmassa/object-detection.torch
# https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/py_cpu_nms.py
# https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
def nms(boxes, scores, overlap=0.5, top_k=0, soft=False, conf_thresh=0, soft_nms_cut=1):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    if boxes.size(-1) == 8:
        # rbox
        return nms_rboxes(boxes, scores, overlap, top_k)

    #assert boxes.size(-1) == 4

    sorted_scores, idx = scores.sort(0)  # sort in ascending order
    if top_k > 0:
        idx = idx[-top_k:]  # indices of the top-k largest vals
        sorted_scores = sorted_scores[-top_k:]

    if not soft:
        sorted_boxes = boxes[idx]
        keep = torchvision_nms(sorted_boxes, sorted_scores, overlap)
        keep = idx[keep]
        return keep, scores[keep]

    keep = []
    new_scores = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    while idx.size(0) > 1:
        i = idx[-1].item()  # index of current largest val
        keep.append(i)

        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i].item())
        yy1 = torch.clamp(yy1, min=y1[i].item())
        xx2 = torch.clamp(xx2, max=x2[i].item())
        yy2 = torch.clamp(yy2, max=y2[i].item())
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        # check sizes of xx1 and xx2.. after each iteration
        inter = w*h

        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        if soft:
            new_scores.append(sorted_scores[-1].item())
            sorted_scores = sorted_scores[:-1]  # remove kept element from view
            # decay the scores

            weights = torch.clamp((soft_nms_cut - IoU) / (soft_nms_cut - overlap), min=0, max=1)
            #weights = 1 - IoU * IoU.ge(overlap).float()
            #weights = torch.exp(- (IoU * IoU) / ((1 - overlap) ** 2))

            sorted_scores *= weights
            # remove scores are too low for saving computation
            high_scores = sorted_scores.ge(conf_thresh)
            idx = idx[high_scores]
            sorted_scores = sorted_scores[high_scores]
        else:
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]

    if idx.size(0) > 0:
        keep.append(idx[-1].item())
        new_scores.append(sorted_scores[-1].item())

    if soft:
        scores = scores.new(new_scores)
    else:
        scores = scores[keep]

    return keep, scores


# Compute bounding box voting
# paper: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Gidaris_Object_Detection_via_ICCV_2015_paper.pdf
# reference https://github.com/sanghoon/pva-faster-rcnn/blob/master/lib/utils/bbox.pyx
def bbox_vote(bbox_nms, score_nms, bbox_all, score_all, thresh):
    bbox_voted = torch.zeros_like(bbox_nms)

    x1 = bbox_all[:, 0]
    y1 = bbox_all[:, 1]
    x2 = bbox_all[:, 2]
    y2 = bbox_all[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    for i, bbox in enumerate(bbox_nms):
        nms_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        xx1 = torch.clamp(x1, min=bbox[0].item())
        yy1 = torch.clamp(y1, min=bbox[1].item())
        xx2 = torch.clamp(x2, max=bbox[2].item())
        yy2 = torch.clamp(y2, max=bbox[3].item())
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)

        inter = w * h

        union = area + nms_area - inter
        IoU = inter / union

        vote = IoU.ge(thresh)
        vote_box = bbox_all[vote]
        vote_score = score_all[vote]
        bbox_voted[i] = torch.sum(vote_box * vote_score.unsqueeze(-1), dim=0) / torch.sum(vote_score)

    return bbox_voted
