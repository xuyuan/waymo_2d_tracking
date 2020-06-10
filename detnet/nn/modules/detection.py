import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.box_utils import decode, nms, bbox_vote, point_form, center_size


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a max number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, variance, top_k=4000, soft_nms=False, bbox_voting=False, class_activation='softmax'):
        super(Detect, self).__init__()
        self.top_k = top_k
        self.variance = variance
        self.soft_nms = soft_nms
        self.bbox_voting = bbox_voting
        self.class_activation = class_activation
        self.kwargs = {}

    def detect(self, loc_data, conf_data, prior_data, conf_thresh, nms_thresh, **kwargs):
        self.kwargs = kwargs
        output = self(loc_data, conf_data, prior_data, conf_thresh, nms_thresh)
        self.kwargs = {}
        return output

    def forward(self, loc_data, conf_data, prior_data, conf_thresh, nms_thresh):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, 4, num_priors]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_classes, num_priors]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        Returns:
            [batch, num_classes, top_k, 5]
        """
        # parameters
        top_k = self.kwargs.get('top_k', self.top_k)
        soft_nms = self.kwargs.get('soft_nms', self.soft_nms)
        bbox_voting = self.kwargs.get('bbox_voting', self.bbox_voting)

        num = loc_data.size(0)  # batch size

        if self.class_activation == 'sigmoid':
            conf_preds = torch.sigmoid(conf_data)
        else:
            # default to softmax
            conf_preds = F.softmax(conf_data, dim=1)
            conf_preds = conf_preds[:, 1:]  # ignore background

        num_classes = conf_preds.size(1)
        output = conf_preds.new_zeros(num, num_classes, top_k, 1 + loc_data.size(1))

        conf_mask = conf_preds.ge(conf_thresh)
        conf_mask_any = conf_mask.any(dim=-1)  # batch x classes
        for i, cl in conf_mask_any.nonzero():
            c_mask = conf_mask[i, cl]

            boxes = decode(loc_data[i, :, c_mask].t(), prior_data[c_mask], self.variance)
            scores = conf_preds[i, cl, c_mask]

            # idx of highest scoring and non-overlapping boxes per class
            boxes_points = point_form(boxes)
            ids, new_scores = nms(boxes_points, scores, nms_thresh, top_k, soft_nms, conf_thresh)

            if bbox_voting > 0:
                new_bbox = bbox_vote(boxes_points[ids], new_scores, boxes_points, scores, bbox_voting)
                new_bbox = center_size(new_bbox)
            else:
                new_bbox = boxes[ids]

            new_scores = new_scores.unsqueeze(1).to(dtype=output.dtype)
            new_bbox = new_bbox.to(dtype=output.dtype)
            torch.cat((new_scores, new_bbox), dim=1, out=output[i, cl, :len(ids)])
        return output
