from PIL.ImageFilter import *
from torchvision.transforms import ColorJitter

from .test_transforms import torchvision_mean
from ..utils.box_utils import encode, match, point_form, center_size, rbox_2_bbox
from ..utils.rbox_utils import generate_rbox_from_masks
from ..trainer.transforms import *
from ..trainer.transforms.vision import *
from ..trainer.transforms.auto_aug import *

from ..nn.point_det import get_heatmap, BoxList


class TransformOnName(object):
    """apply transform to one given name only"""

    def __init__(self, transform, key_name='input'):
        self.transform = transform
        self.key_name = key_name

    def __call__(self, sample):
        if self.key_name in sample:
            image = sample[self.key_name]
            sample[self.key_name] = self.transform(image)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '({0}: {1})'.format(self.key_name, self.transform)


class TrainAugmentation(Compose):
    def __init__(self, size, bg_color='black',
                 crop_remove_small_box=True,
                 truncate_box=True,
                 random_pad_max_ratio=0.0,
                 cut_out=True,
                 min_scale=0.9,
                 max_scale=1.1,
                 cut_mix_dataset=None):
        BG_COLORS = dict(black=(0, 0, 0), white=(255, 255, 255), gray=TORCH_VISION_MEAN * 255)
        bg_color = BG_COLORS[bg_color]

        transforms = []
        if random_pad_max_ratio > 0:
            max_padding = tuple(s * random_pad_max_ratio for s in size)
            transforms += [RandomApply([RandomPad(max_padding, bg_color)])]

        transforms += [TryApply(RandomCrop(min_size=np.asarray(size) * min_scale,
                                           max_size=np.asarray(size) * max_scale,
                                           remove_bbox_outside=crop_remove_small_box,
                                           truncate_bbox=truncate_box,
                                           focus=False),
                                max_trail=5, fallback_origin=True)]

        transforms.append(Resize(size))

        transforms += [RandomApply([HorizontalFlip()])]
        #
        # transforms += [RandomApply(RandomCopyPaste(1/4, 1/2, cut_mix_dataset))]

        if cut_out:
            cut_out = [RandomApply(TryApply(CutOut(max_size=max(size) // 4, fill=i))) for i in (None, bg_color)]
            transforms += cut_out

        color_jitter = RandomChoice([ColorJitter(brightness=0.25, contrast=0.5),
                                     AutoContrast(),
                                     # Equalize(),
                                     CLAHE()
                                     ])

        noise = RandomChoice([GaussNoise(0.1),
                              SaltAndPepper(),
                              RandomSharpness(),
                              JpegCompression(),
                              RandomChoice([ImageFilter(BLUR),
                                            ImageFilter(DETAIL),
                                            ImageFilter(ModeFilter(size=3)),
                                            ImageFilter(GaussianBlur()),
                                            ImageFilter(MaxFilter(size=3)),
                                            ImageFilter(MedianFilter(size=3))])])

        # deform
        deform = RandomChoice([#RandomSkew(0.1),
                               RandmonRotate(5),
                               # GridDistortion(),
                               # ElasticDeformation(approximate=True),
                               ])

        transforms.append(RandomApply([RandomChoice([
            color_jitter,
            noise,
            deform,
            COCOAugment(),
        ])]))

        super().__init__(transforms)


class MatchPriorBox(object):
    def __init__(self, image_size, priors, variance, overlap_bg_threshold, overlap_fg_threshold, return_iou=False,
                 warn_unmatched_bbox=False, dynamic_bg_threshold=None):
        self.return_iou = return_iou
        assert overlap_bg_threshold <= overlap_fg_threshold
        self.overlap_fg_threshold = overlap_fg_threshold
        self.overlap_bg_threshold = overlap_bg_threshold
        self.dynamic_bg_threshold = dynamic_bg_threshold

        self.variance = variance
        self.defaults = priors.cpu()
        self.defaults_point = point_form(self.defaults)
        # clamp points for bbox on the border
        torch.clamp(self.defaults_point[:, 0], min=0, max=image_size[1], out=self.defaults_point[:, 0])
        torch.clamp(self.defaults_point[:, 1], min=0, max=image_size[0], out=self.defaults_point[:, 1])
        torch.clamp(self.defaults_point[:, 2], min=0, max=image_size[1], out=self.defaults_point[:, 2])
        torch.clamp(self.defaults_point[:, 3], min=0, max=image_size[0], out=self.defaults_point[:, 3])
        self.defaults = center_size(self.defaults_point)
        self.defaults_areas = self.defaults[:, 2] * self.defaults[:, 3]

        if len(self.variance) == 2:
            self.defaults_bounds = self.defaults_point
        else:
            assert self.defaults_point.size(-1) == 8  # (-1, 8)
            self.defaults_bounds = rbox_2_bbox(self.defaults_point)

        self.has_warn = not warn_unmatched_bbox

    def __call__(self, sample):
        defaults = self.defaults
        num_priors = defaults.size(0)

        if len(self.variance) == 2:
            bboxes = sample['bbox'][:, :4]
            bboxes = torch.from_numpy(bboxes).float()
            bboxes_point = bboxes.to(defaults.device)  # point form
            if len(bboxes) > 0:
                boxes = center_size(bboxes_point)  # offset form
            else:
                boxes = bboxes_point
            boxes_point = bboxes_point
        elif len(self.variance) == 3:
            if 'rboxes' not in sample:
                # compute rboxes from masks
                rboxes = generate_rbox_from_masks(sample['masks'], len(sample['bbox']))
                sample['rboxes'] = rboxes

            boxes = torch.from_numpy(sample['rboxes']).float()  # offset form
            if len(boxes) > 0:
                boxes_point = point_form(boxes)  # point form
                bboxes_point = rbox_2_bbox(boxes_point)
            else:
                boxes_point = np.empty((0, 4, 2))
                bboxes_point = np.empty((0, 4))
        else:
            raise NotImplementedError(self.variance)

        if len(boxes) == 0:
            priors_loc = defaults.new_zeros((0, defaults.shape[1]))
            priors_label = defaults.new_zeros(num_priors, dtype=torch.long)
            priors_pos = defaults.new_zeros(num_priors, dtype=torch.bool)
            priors_iou = defaults.new_zeros(num_priors, dtype=torch.float32)

            # if not self.has_warn:
            #     self.has_warn = True
            #     warnings.warn(f"image {sample.get('image_id', '?')} without any object")
        else:
            labels = sample['bbox'][:, -1].astype(int)
            match_idx, priors_iou = match(boxes, boxes_point, bboxes_point,
                                          self.defaults, self.defaults_point, self.defaults_bounds, self.defaults_areas)
            priors_label = torch.from_numpy(labels[match_idx])  # [num_priors] top class label for each prior

            overlap_bg_threshold = self.overlap_bg_threshold
            if self.dynamic_bg_threshold is not None:
                overlap_bg_threshold = self.overlap_fg_threshold
                for idx in range(len(bboxes)):
                    idx_iou = priors_iou[match_idx == idx]
                    if len(idx_iou) > 0:
                        overlap_bg_threshold = min(overlap_bg_threshold, idx_iou.max())
                overlap_bg_threshold -= self.dynamic_bg_threshold
                overlap_bg_threshold = max(self.overlap_bg_threshold, overlap_bg_threshold)

            priors_index = None
            foreground_threshold = self.overlap_bg_threshold if self.return_iou else self.overlap_fg_threshold
            priors_pos = priors_iou >= foreground_threshold  # foreground
            if priors_pos.any():
                priors_index = match_idx[priors_pos]

            # relax threshold to get more positive anchors
            # threshold_delta = 0.1
            # n_boxes = len(match_idx.unique())
            # while foreground_threshold >= (self.overlap_bg_threshold + threshold_delta) and (priors_index is None or len(torch.unique(priors_index)) < n_boxes):
            #     foreground_threshold -= threshold_delta
            #     priors_pos = priors_iou > foreground_threshold
            #     if priors_pos.any():
            #         priors_index = match_idx[priors_pos]

            priors_label[priors_iou < foreground_threshold] = -100  # ignore_index
            neg = priors_iou < overlap_bg_threshold  # background
            priors_label[neg] = 0  # set background

            if priors_pos.any():
                priors_bbox = boxes[priors_index]  # [num_pos, 4] offset form
                priors_loc = encode(priors_bbox, defaults[priors_pos, :], self.variance)  # [num_pos,4] offset form
            else:
                priors_loc = defaults.new_zeros((0, defaults.shape[1]))

            if not self.has_warn:
                # check not used GT
                used_bbox_idx = match_idx.unique()
                unused_bbox = []
                unused_label = []
                for i in range(len(boxes)):
                    if i not in used_bbox_idx:
                        area = boxes[i][2] * boxes[i][3]
                        if area > 1:  # ignore tiny box
                            unused_bbox.append(boxes[i])
                            unused_label.append(labels[i])
                image_h, image_w = sample['input'].shape[-2:]
                umatched_bbox = [(b, l) for b, l in zip(unused_bbox, unused_label) if
                                 0 < b[0] < image_w and 0 < b[1] < image_h]

                if umatched_bbox:
                    self.has_warn = True
                    warnings.warn('{} out of {} bbox are unmatched in {}: {}'.format(len(umatched_bbox), len(boxes),
                                                                                     sample['image_id'], umatched_bbox))

        sample_ground_truth = dict(priors_label=priors_label,
                                   priors_loc=priors_loc,
                                   priors_pos=priors_pos)
        if self.return_iou:
            weights_iou = torch.ones_like(priors_iou)
            soft = (priors_iou < self.overlap_fg_threshold) & (priors_iou >= overlap_bg_threshold)
            weights_iou[soft] = ((priors_iou[soft] - overlap_bg_threshold) / (
                        self.overlap_fg_threshold - overlap_bg_threshold)).clamp(0, 1)
            sample_ground_truth['weights_iou'] = weights_iou

        sample_ground_truth.update(sample)
        return sample_ground_truth

    def __repr__(self):
        format_string = self.__class__.__name__ + '(n={}, variance={})'.format(self.defaults.size(0),
                                                                               self.variance)
        return format_string


def create_match_prior_box(ssd_net, image_size, overlap_bg_threshold, overlap_thresh, return_iou, warn_unmatched_bbox):
    # call forward for computing priorbox
    with torch.no_grad():
        dummy_image = torch.zeros((1, 3, image_size[0], image_size[1]), dtype=torch.float)
        ssd_net.eval()
        ssd_net(dummy_image)
    mpb = MatchPriorBox(image_size, ssd_net.priorbox.priors, ssd_net.detector.variance,
                        overlap_bg_threshold, overlap_thresh, return_iou, warn_unmatched_bbox)
    return mpb


class PointHeatMap(object):
    def __init__(self, scale, num_class):
        self.scale = scale
        self.num_class = num_class

    def __call__(self, sample):
        bbox = sample['bbox'][:, :4]
        labels = sample['bbox'][:, 4].astype(np.int) - 1

        h, w = sample['input'].shape[-2:]
        gt_list = BoxList(bbox, (w, h))
        gt_list.add_field('labels', labels)
        gt_list.resize((w // self.scale, h // self.scale))
        heatmap = get_heatmap(gt_list, 0, [self.scale], radius=1, class_num=self.num_class)
        return dict(input=sample['input'], heatmap=heatmap)


def creat_point_heat_map(point_det, image_size):
    with torch.no_grad():
        dummy_image = torch.zeros((1, 3, image_size[0], image_size[1]), dtype=torch.float)
        point_det.eval()
        heatmap = point_det(dummy_image)
        h, w = heatmap.shape[-2:]
        scale_h = image_size[0] // h
        scale_w = image_size[1] // w
        assert scale_h == scale_w
        return PointHeatMap(scale_h, point_det.num_class)
