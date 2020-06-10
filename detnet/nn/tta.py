
import torch
from torch import nn
import numpy as np
from ..utils.box_utils import jaccard_bbox, point_form, nms, center_size


def nms_detections(detections, iou_thresh=0.5, soft=False, soft_nms_cut=1):
    scores = [bbox[:, 0:1] for bbox in detections]
    bboxes = [bbox[:, 1:5] for bbox in detections]
    scores = torch.from_numpy(np.vstack(scores).flatten())
    bboxes = torch.from_numpy(np.vstack(bboxes))
    bboxes = point_form(bboxes)

    keep, scores = nms(bboxes, scores, overlap=iou_thresh, soft=soft, soft_nms_cut=soft_nms_cut)
    bboxes = bboxes[keep]
    bboxes = center_size(bboxes)
    bboxes = torch.cat((scores[:, None], bboxes), dim=1)
    return bboxes.numpy()


def merge_detections(detections, nms_thresh=0.5):
    """

    Parameters
    ----------
    detections  N_TTA x N_BOX x [score, cx, cy, w, h]
    conf_thresh
    nms_thresh

    Returns
    -------
    """
    results = detections[0]
    results[..., 0] /= len(detections)
    results[..., 1:] *= results[..., :1]  # weighted coordinate by conf

    for others in detections[1:]:
        if len(others) > 0:
            others[..., 0] /= len(detections)
            others[..., 1:] *= others[..., :1]  # weighted coordinate by conf

            if len(results) > 0:
                m_box = results[:, 1:] / results[:, :1]  # weighted to absolute coordinate
                o_box = others[:, 1:] / others[:, :1]
                overlaps = jaccard_bbox(torch.from_numpy(m_box), torch.from_numpy(o_box))
                iou, idx = overlaps.max(0)

                # matched only
                matched = (iou >= nms_thresh).numpy()
                if matched.any():
                    idx_matched = idx[matched]
                    o_matched = others[matched]
                    results_matched = results[idx_matched]
                    results[idx_matched] += o_matched.reshape(results_matched.shape)

                unmatched = (iou < nms_thresh).numpy()
                if unmatched.any():
                    results = np.vstack((results, others[unmatched]))
            else:
                # copy
                results = others

    # weighted to absolute coordinate
    results[..., 1:] /= results[..., :1]
    return results


class OpTTA(object):
    def pre_process(self, x):
        """
        Args:
            x: [[N, C, H, W], ...]

        Returns:
            [[N, C, H, W], ...] * K
        """
        raise NotImplementedError

    def post_process(self, y):
        """
        Args:
            y: [C, [n, 5]] * K

        Returns:
            [C, [n, 5]]
        """
        raise NotImplementedError

    def pprint_tree(self, file=None, _prefix="", _last=True):
        sub_tree = isinstance(self, Compose)
        node_value = '+' if sub_tree else str(self)
        print(_prefix, "`- " if _last else "|- ", node_value, sep="", file=file)
        _prefix += "   " if _last else "|  "

        if sub_tree:
            child_count = len(self)
            for i, child in enumerate(self):
                _last = i == (child_count - 1)
                child.pprint_tree(file, _prefix, _last)


class Compose(OpTTA, list):
    pass


class SequentialTTA(Compose):
    def pre_process(self, x):
        for tta in self:
            x = tta.pre_process(x)
        return x

    def post_process(self, y):
        for tta in self[::-1]:
            y = tta.post_process(y)
        return y


class ParallelTTA(Compose):
    def __init__(self, ttas, merge_func=merge_detections):
        super().__init__(ttas)
        self.merge_func = merge_func

    def pre_process(self, x):
        X = [tta.pre_process(x) for tta in self]
        return sum(X, [])

    def post_process(self, y):
        l = len(y) // len(self)
        Y = [y[i*l:i*l+l] for i in range(len(self))]
        Y = [tta.post_process(yi) for yi, tta in zip(Y, self)]

        return [[[self.merge_func(b)  # n*4
                  for b in zip(*ci)]  # C, n*4
                 for ci in zip(*yi)]  # N, C, n*4
                for yi in zip(*Y)]   # [N, C, n*4]


class OrigTTA(OpTTA):
    def pre_process(self, x):
        return x

    def post_process(self, y):
        return y


class HFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.flip(xi, [3]) for xi in x]

    def post_process(self, y):
        for yi in y:
            for d in yi:
                for c in d:
                    c[..., 1] = 1 - c[..., 1]
        return y


class VFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.flip(xi, [2]) for xi in x]

    def post_process(self, y):
        for yi in y:
            for d in yi:
                for c in d:
                    c[..., 2] = 1 - c[..., 2]
        return y


class DFlipTTA(OpTTA):
    def pre_process(self, x):
        return [torch.transpose(xi, 2, 3) for xi in x]

    def post_process(self, y):
        return [[[c[:, [0, 2, 1, 4, 3]] for c in d] for d in yi] for yi in y]


class ResizeTTA(OpTTA):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'

    def pre_process(self, x):
        return [torch.nn.functional.interpolate(xi, scale_factor=self.scale_factor, mode='bilinear',  align_corners=False) for xi in x]

    def post_process(self, y):
        return y


class BatchTTA(OpTTA):
    def pre_process(self, x):
        batch = []
        self._indices = []
        for b in self._batch(x):
            if len(b) > 0:
                l = [len(bi) for bi in b]
                self._indices.append(l)

                b = torch.cat(b, dim=0)
                batch.append(b)
        return batch

    def post_process(self, y):  # [N, C, n*5]
        z = []
        for l, yi in zip(self._indices, y):
            i = 0
            for j in l:
                z.append(yi[i:i+j])
                i += j
        return z

    def _batch(self, x):
        b = x[:1]
        h, w = x[0].shape[-2:]
        for xi in x[1:]:
            if xi.shape[-1] != w or xi.shape[-2] != h:
                yield b
                b = [xi]
                h, w = b[0].shape[-2:]
            else:
                b.append(xi)
        yield b


class TTA(nn.Module):
    def __init__(self, detector, data_aug):
        super().__init__()
        self.detector = detector

        ttas = []
        brute_mode = 'brute' in data_aug

        if 'orig' in data_aug:
            ttas.append(OrigTTA())

        for aug in data_aug:
            if aug.startswith('x'):
                scale_factor = float(aug[1:])
                ttas.append(ResizeTTA(scale_factor))

        if brute_mode and len(ttas) > 1:
            ttas = [ParallelTTA(ttas)]

        if 'dflip' in data_aug:
            ttas.append(DFlipTTA())
        if 'hflip' in data_aug:
            ttas.append(HFlipTTA())
        if 'vflip' in data_aug:
            ttas.append(VFlipTTA())

        if brute_mode and len(ttas) > 1:
            ttas = ttas[:1] + [ParallelTTA([OrigTTA(), tta]) for tta in ttas[1:]]

        if 'batch' in data_aug:
            ttas.append(BatchTTA())
        self.tta = SequentialTTA(ttas)

        self.tta.pprint_tree()

    def predict(self, x):
        X = self.tta.pre_process([x])
        Y = [self.detector.predict(xi) for xi in X]
        y = self.tta.post_process(Y)
        return y[0]


class _TTA(nn.Module):
    def __init__(self, detector, data_aug):
        super().__init__()
        self.detector = detector
        self.data_aug = data_aug

    def predict(self, x):
        batch_mode = False
        if x.dim() == 4:
            assert x.shape[0] == 1  # batch size == 1
            x = x[0]
            batch_mode = True

        x = self.preprocess(x)
        y = self.detector.predict(x)
        z = [[yi[c] for yi in y] for c in range(len(y[0]))]
        z = [self.post_process(zc) for zc in z]

        if batch_mode:
            z = [z]
        return z

    def preprocess(self, x):
        batch = []

        if 'orig' in self.data_aug:
            batch.append(x)

        if 'hflip' in self.data_aug:
            batch_no_flip = batch if batch else [x]
            x_hflip = [torch.flip(x, [2]) for x in batch_no_flip]
            batch += x_hflip

        if 'vflip' in self.data_aug:
            batch_no_flip = batch if batch else [x]
            x_vflip = [torch.flip(x, [1]) for x in batch_no_flip]
            batch += x_vflip

        if 'dflip' in self.data_aug:
            assert x.shape[-1] == x.shape[-2]
            batch_no_flip = batch if batch else [x]
            x_dflip = [torch.transpose(x, 1, 2) for x in batch_no_flip]
            batch += x_dflip

        batch = torch.stack(batch)
        return batch

    def _hflip_bboxes(self, y):
        y[..., 1] = 1 - y[..., 1]

    def _vflip_bboxes(self, y):
        y[..., 2] = 1 - y[..., 2]

    def _transpose_bboxes(self, y):
        y[:] = y[:, [0, 2, 1, 4, 3]]

    def post_process(self, y):
        end = len(y)
        n = end
        if 'dflip' in self.data_aug:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    self._transpose_bboxes(y[j])
            n = n // 2

        if 'vflip' in self.data_aug:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    self._vflip_bboxes(y[j])
            n = n // 2

        if 'hflip' in self.data_aug:
            for i in range(0, end, n):
                for j in range(i + n // 2, i + n):
                    self._hflip_bboxes(y[j])

        return merge_detections(y)
