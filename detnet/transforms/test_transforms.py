import warnings
import math

import torch
import torchvision
import torch.nn.functional as F
from PIL import ImageOps
from PIL.Image import Image

try:
    import skimage
    from skimage.morphology import watershed, dilation, square, erosion
    from skimage.exposure import equalize_adapthist
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_otsu
    import scipy.ndimage as ndi
    from ..utils.rbox_utils import optimize_rbox_mask
except Exception as e:
    pass

from ..utils.box_utils import jaccard_bbox

import numpy as np

# normalization for `torchvision.models` see http://pytorch.org/docs/master/torchvision/models.html
torchvision_mean = [0.485, 0.456, 0.406]
torchvision_std = [0.229, 0.224, 0.225]


class Preprocessor(object):
    def __init__(self, min_dim, mean, std):
        self.min_dim = int(min_dim * 1)
        self.mean = mean
        self.std = std
        self.size_divisible = 32
        self.data_aug = None

        self.min_dim += self.divisible_padding(self.min_dim)

    def divisible_padding(self, size):
        """return size to pad for divisible"""
        if self.size_divisible > 1:
            return int(math.ceil(size / self.size_divisible) * self.size_divisible) - size
        return 0

    def _pic_to_input(self, pic):
        x = torchvision.transforms.functional.to_tensor(pic)
        # somehow this is slow!
        # x = torchvision.transforms.functional.normalize(x, self.mean, self.std)
        return x

    def __call__(self, pil_image, min_dim=None, data_aug=None):
        if min_dim is None:
            min_dim = self.min_dim

        height_width = (pil_image.height, pil_image.width)

        x = self._pic_to_input(pil_image)
        batch = []

        if not data_aug:
            data_aug = self.data_aug

        if data_aug:
            if 'orig' in data_aug:
                batch.append(x)

            if 'autocontrast' in data_aug:
                ac_pic = ImageOps.autocontrast(pil_image)
                batch.append(self._pic_to_input(ac_pic))

            if 'equalize' in data_aug:
                eq_pic = equalize_adapthist(np.asarray(pil_image)).astype(np.float32)
                batch.append(self._pic_to_input(eq_pic))

            if 'hflip' in data_aug:
                batch_no_flip = batch if batch else [x]
                x_hflip = [torch.flip(x, [2]) for x in batch_no_flip]
                batch += x_hflip

            if 'vflip' in data_aug:
                batch_no_flip = batch if batch else [x]
                x_vflip = [torch.flip(x, [1]) for x in batch_no_flip]
                batch += x_vflip

            if 'dflip' in data_aug:
                # assume width == height
                batch_no_flip = batch if batch else [x]
                x_dflip = [torch.transpose(x, 1, 2) for x in batch_no_flip]
                batch += x_dflip
        else:
            batch.append(x)

        batch = torch.stack(batch)

        # make sure image size meets min_dim
        pad_bottom = max(height_width[0], min_dim) - height_width[0]
        pad_right = max(height_width[1], min_dim) - height_width[1]
        if self.size_divisible > 1:
            pad_bottom += self.divisible_padding(height_width[0] + pad_bottom)
            pad_right += self.divisible_padding(height_width[1] + pad_right)
        if pad_bottom > 0 or pad_right > 0:
            batch = F.pad(batch, [0, pad_right, 0, pad_bottom])

        # Input norm
        for i in range(batch.shape[1]):
            batch[:, i, :, :] -= self.mean[i]
            batch[:, i, :, :] /= self.std[i]
        return batch, height_width


class TestTransform(object):
    def __init__(self, min_dim, mean, std):
        self.preprocessor = Preprocessor(min_dim, mean, std)
        self.batch_squeezed = False

    def pre_process(self, data, min_dim, data_aug):
        self.batch_squeezed = False
        if isinstance(data, Image):
            return self.preprocessor(data, min_dim=min_dim, data_aug=data_aug)
        elif isinstance(data, list):
            # image has been preprocessed already
            data, height_width = data
            height_width = (height_width[0].item(), height_width[1].item())
            assert torch.is_tensor(data)

            if data.dim() > 3:
                self.batch_squeezed = True

            if data.dim() > 4:
                data.squeeze_(dim=0)
            return data, height_width

        elif torch.is_tensor(data):
            if data.dim() > 3:
                self.batch_squeezed = True

            if data.dim() > 4:
                data.squeeze_(dim=0)

            height_width = (data.shape[-2], data.shape[-1])
            return data, height_width
        else:
            print(data)
            raise NotImplementedError(type(data))

    def _hflip_bboxes(self, y, width):
        y[..., 1] = width - y[..., 1]

    def _vflip_bboxes(self, y, height):
        y[..., 2] = height - y[..., 2]

    def _transpose_bboxes(self, y):
        y[:] = y[:, :, [0, 2, 1, 4, 3]]

    def post_process(self, height_width, y, data_aug, conf_thresh, nms_thresh, max_bbox=0):
        height, width = height_width
        if data_aug:
            end = len(y)
            n = end
            if 'dflip' in data_aug:
                assert width == height
                for i in range(0, end, n):
                    for j in range(i + n // 2, i + n):
                        self._transpose_bboxes(y[j])
                n = n // 2

            if 'vflip' in data_aug:
                for i in range(0, end, n):
                    for j in range(i + n // 2, i + n):
                        self._vflip_bboxes(y[j], height)
                n = n // 2

            if 'hflip' in data_aug:
                for i in range(0, end, n):
                    for j in range(i + n // 2, i + n):
                        self._hflip_bboxes(y[j], width)

        y = self.merge_detections(y, conf_thresh / len(y) / 2, nms_thresh)

        scale_height = 1 / height
        scale_width = 1 / width
        scale = y.new([scale_width, scale_height, scale_width, scale_height])
        y[:, :, 1:5] *= scale

        y[:, :, 1:5].clamp_(0, 1)
        wh = y[:, :, 3] * y[:, :, 4]

        if max_bbox > 0:
            all_conf = y[:, :, 0].view(-1)
            top_conf, _ = all_conf.topk(max_bbox)
            conf_thresh = max(top_conf[-1].item(), conf_thresh)

        detections = []
        valid_y = (y[:, :, 0] >= conf_thresh) & (wh > scale_height * scale_width)
        for i in range(y.size(0)):
            detection_i = y[i, valid_y[i]]
            detections.append(detection_i)

        if self.batch_squeezed:
            detections = [detections]
        return detections

    def merge_detections(self, detections, conf_thresh, nms_thresh):
        """

        Parameters
        ----------
        detections  N_TTA x N_CLS x N_BOX x 4(5)
        conf_thresh
        nms_thresh

        Returns
        -------
        """
        if len(detections) == 1:
            return detections[0]

        detections[:, :, :, 0] /= detections.size(0)
        detections[:, :, :, 1:] *= detections[:, :, :, :1]  # weighted coordinate by conf

        results = detections[0]

        for others in detections[1:]:
            for i in range(results.size(0)):  # for each class
                # assume conf in sorted
                o = others[i]
                o_num = (o[:, 0] > conf_thresh).sum()
                if o_num == 0:
                    continue

                m = results[i]
                m_num = (m[:, 0] > conf_thresh).sum()
                if m_num > 0:
                    m_box = m[:m_num, 1:] / m[:m_num, :1]  # weighted to absolute coordinate
                    o_box = o[:o_num, 1:] / o[:o_num, :1]
                    overlaps = jaccard_bbox(m_box, o_box)
                    iou, idx = overlaps.max(0)

                    # matched only
                    matched = iou >= nms_thresh
                    idx_matched = idx[matched]
                    o_matched = o[:o_num][matched]
                    m[idx_matched] += o_matched

                    unmatched = o[:o_num][iou < nms_thresh]
                    max_unmatched = len(m) - m_num
                    if max_unmatched > 0:
                        unmatched = unmatched[:max_unmatched]
                        m[m_num:m_num+len(unmatched)] = unmatched
                else:
                    # copy all
                    results[i] = others[i]

        # weighted to absolute coordinate
        conf = results[:, :, :1]
        conf = torch.where(conf > 0, conf, torch.ones_like(conf))
        results[:, :, 1:] /= conf
        return results

    def post_process_mask(self, height_width, mask, bboxes, data_aug, conf_thresh, nms_thresh, mask_thresh, mask2rbox):
        mask = F.softmax(mask, dim=1)
        if data_aug:
            mask = self.decode_augmented_mask(mask, data_aug)
        else:
            mask = mask[0]
        bboxes = self.post_process(height_width, bboxes, data_aug, conf_thresh, nms_thresh)
        segmentation = []
        #from matplotlib import pyplot as plt
        # plt.imshow(mask[0])
        # plt.show()
        #plt.imshow(mask[1])
        #plt.show()
        #mask = (mask[1:] >= 0.5).cpu().int().numpy()
        for detection, segm in zip(bboxes, mask[1:].cpu().numpy()):
            seg = np.zeros_like(segm, dtype=np.int)
            height, width = seg.shape
            scale = np.asarray([width, height])
            for j in range(len(detection) - 1, -1, -1):
                # assume detection are sorted by confidence, so putting segmentation from low confidence
                det = detection[j]

                # to points form
                center = det[1:3]
                half_wh = det[3:5] * 0.5
                x0, y0 = ((center - half_wh) * scale).astype(int)
                x1, y1 = ((center + half_wh) * scale).astype(int) + 1
                x0 = np.clip(x0, 0, width)
                x1 = np.clip(x1, 1, width + 1)
                y0 = np.clip(y0, 0, height)
                y1 = np.clip(y1, 1, height + 1)

                seg_j = segm[y0:y1, x0:x1]
                if seg_j.size:
                    # from skimage.filters import try_all_threshold
                    # from matplotlib import pyplot as plt
                    # plt.imshow(seg_j)
                    # plt.show()
                    # try:
                    #     fig, ax = try_all_threshold(seg_j, figsize=(10, 8), verbose=False)
                    #     plt.show()
                    # except Exception as e:
                    #     print(e)

                    if mask_thresh < 0:
                        try:
                            thr_m = threshold_otsu(seg_j)
                        except Exception as e:
                            warnings.warn(str(e))
                            min_m, max_m = np.min(seg_j), np.max(seg_j)
                            thr_m = (min_m + max_m) / 2
                    else:
                        thr_m = mask_thresh
                    seg_j = (seg_j >= thr_m)
                    if mask2rbox:
                        seg_j = optimize_rbox_mask(seg_j, 1)
                    seg[y0:y1, x0:x1] = seg_j * (j+1) + (1 - seg_j) * seg[y0:y1, x0:x1]

            segmentation.append(seg)
        return (bboxes, segmentation)

    def post_process_mask_with_elevation(self, height_width, mask, elevation, bboxes, data_aug, conf_thresh, nms_thresh):
        mask = F.softmax(mask, dim=1)
        if data_aug:
            mask = self.decode_augmented_mask(mask, data_aug)
            elevation = self.decode_augmented_mask(elevation, data_aug)
        else:
            mask = mask[0]
            elevation = elevation[0]

        bboxes = self.post_process(height_width, bboxes, data_aug, conf_thresh, nms_thresh)

        mask_array = mask.data.cpu().numpy()
        elevation_array = elevation.data.cpu().numpy()
        elevation = elevation_array[0]

        # watershed on elevation
        markers = np.zeros_like(elevation)
        markers[elevation <= 0] = 1
        markers[elevation > 0.5] = 2

        segmentation = watershed(elevation, markers)
        segmentation = binary_fill_holes(segmentation - 1)
        segmentation = dilation(segmentation, square(3))
        instances, _ = ndi.label(segmentation)
        elevation0 = np.zeros_like(elevation)

        instances_segmentation = []
        for prob, detection in zip(mask_array[1:], bboxes):
            prob_mask = prob > 0.5
            if prob_mask.any():
                labeled_instance = watershed(elevation0, instances, mask=prob_mask)
                labeled_instance = labeled_instance * prob_mask
            else:
                labeled_instance = elevation0

            # labeled_nuclei3 = np.zeros_like(labeled_nuclei2)
            # for j, det in enumerate(detection):
            #    score, pt = det
            #    x0, y0, x1, y1 = pt.astype(int)
            #    labeled_nuclei3[y0:y1, x0:x1] = labeled_nuclei2[y0:y1, x0:x1]

            instances_segmentation.append(labeled_instance)

        return bboxes, np.asarray(instances_segmentation)

    def decode_augmented_mask(self, mask, data_aug):
        hflip = 'hflip' in data_aug
        vflip = 'vflip' in data_aug
        dflip = 'dflip' in data_aug

        end = len(mask)
        n = end
        if dflip:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    mask[j] = torch.transpose(mask[j], 1, 2)
            n = n // 2

        if vflip:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    mask[j] = torch.flip(mask[j], [1])
            n = n // 2

        if hflip:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    mask[j] = torch.flip(mask[j], [2])

        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(mask.size(0), mask.size(1))
        # for i in range(mask.size(0)):
        #     for j in range(mask.size(1)):
        #         axes[i, j].imshow(mask[i, j])
        # plt.show()

        return mask.mean(dim=0)

