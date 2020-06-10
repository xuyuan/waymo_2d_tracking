
import warnings
from functools import cmp_to_key
from pathlib import Path
from collections import Iterable
import numpy as np
from PIL import Image
import pandas as pd

from .anchor_box_dataset import AnchorBoxDataset
from ..trainer.data import WeightedRandomDataset


COCO_CATEGORIES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

COCO_CLASSNAMES = [COCO_CATEGORIES.get(i, "unknown") for i in range(1, 91)]


def get_label_map():
    """get label map which ignores unknown classes, e.g. total 80 classes
    """
    label_map = {}
    classnames = []
    for coco_id, coco_name in COCO_CATEGORIES.items():
        label_map[coco_id] = len(classnames)
        classnames.append(coco_name)
    return label_map, classnames


def category_count(cocoGt):
    data = [ann['category_id'] for ann in cocoGt.anns.values()]
    value_counts = pd.Series(data).value_counts()
    return value_counts


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, coco, label_map=None, segmentation=False):
        self.coco = coco
        self.label_map = label_map
        self.segmentation = segmentation

    def __call__(self, target, img):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        boxes = []
        labels = []

        width, height = img.size
        scale = np.asarray([width, height, width, height], dtype=np.float32)

        if self.segmentation:
            masks = np.zeros((height, width), dtype=np.int32)
        else:
            masks = None

        for obj in target:
            if 'bbox' in obj:
                bbox = np.asarray(obj['bbox'], dtype=np.float32)  # (left_top_x, left_top_y, width, height)
                bbox[2:] += bbox[:2]  # (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
            else:
                warnings.warn("no bbox!")
                continue

            label_idx = obj['category_id']
            if self.label_map:
                label_idx = self.label_map[label_idx]

            if self.segmentation:
                if 'segmentation' in obj:
                    mask_obj = self.coco.annToMask(obj)
                    masks[mask_obj > 0] = label_idx
                else:
                    warnings.warn("no segmentation!")

            bbox /= scale  # [xmin, ymin, xmax, ymax] in percentage
            boxes.append(bbox)
            labels.append(label_idx)

        if boxes:
            boxes=np.asarray(boxes)
            labels=np.asarray(labels)
            bbox = np.hstack((boxes, labels[:, None]))
            bbox = bbox[labels > 0]
            if bbox.size > 0:
                bbox = np.unique(bbox, axis=0)
        else:
            bbox = np.empty((0, 5))
        sample = dict(input=img, bbox=bbox)

        if self.segmentation:
            sample['masks'] = masks

        return sample


class COCODetection(AnchorBoxDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2014>`_ Dataset.
    Args:
        images_root (string): Root directory where images are downloaded to.
        annotations (string): annotations file of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """
    metric='COCO'

    def __init__(self, images_root, annotations, dataset_name=None, image_ids=None, return_image_file=False,
                 sort_by_image_size=False,
                 segmentation=False):
        self.export_mode = False
        from pycocotools.coco import COCO
        self.return_image_file = return_image_file
        self.root = Path(images_root)
        self.coco = COCO(annotations)

        if image_ids is None:
            self.ids = sorted(self.coco.getImgIds())
        else:
            self.ids = image_ids
            self.coco.imgs = {k: v for k, v in self.coco.imgs.items() if k in image_ids}

        label_map = None
        self.classnames = None
        if dataset_name == 'MS COCO':
            label_map, self.classnames = get_label_map()

        self.target_transform = None
        if self.coco.anns:
            self.target_transform = COCOAnnotationTransform(self.coco, label_map, segmentation)
        if self.classnames is None:
            cats = self.coco.cats
            cats_ids = [c['id'] for c in cats.values()]
            self.classnames = ['background'] * (max(cats_ids) + 1)
            for c in cats.values():
                self.classnames[c['id']] = c['name']

        if sort_by_image_size:
            self.ids = sorted(self.ids, key=cmp_to_key(self._image_size_compare))

    def exclude(self, image_ids):
        self.ids = [i for i in self.ids if i not in image_ids]

    def set_export_mode(self, mode):
        self.export_mode = mode

    def _image_size_compare(self, x, y):
        x = self._read_image(x)
        y = self._read_image(y)
        if x.height == y.height:
            return x.width - y.width
        return x.height - y.height

    def __len__(self):
        return len(self.ids)

    def _get_image_path(self, img_id):
        return self.root / self.coco.imgs[img_id]['file_name']

    def _read_image(self, img_id):
        path = self._get_image_path(img_id)
        assert path.exists(), 'Image path does not exist: {}'.format(path)
        image = Image.open(path)
        return image

    def _get_target(self, img_id):
        return self.coco.imgToAnns[img_id]

    def getitem(self, index):
        img_id = self.ids[index]
        sample = dict(image_id=str(img_id))

        if not self.export_mode:
            img = self._read_image(img_id)
            if self.target_transform is not None:
                target = self._get_target(img_id)
                target = self.target_transform(target, img)
                sample.update(target)
            else:
                sample['input'] = img

        if self.return_image_file:
            sample['image_file'] = self._get_image_path(img_id)

        return sample

    def __str__(self):
        fmt_str = [self.__class__.__name__,
                   'No. of images: {}'.format(len(self)),
                   'No. of classes: {}'.format(len(self.classnames) - 1),
                   'No. of bboxes {}'.format(len(self.coco.anns)),
                   'Root Location: {}'.format(self.root)]
        fmt_str = ('\n' + self.repr_indent).join(fmt_str)
        return fmt_str + '\n'

    def get_category_id(self, name):
        for cat in self.coco.cats.values():
            if cat['name'] == name:
                return cat['id']
        return 0

    def load_prediction(self, predictions):
        """return list of dict
            {
            "image_id" : str,
            "category_id" : int,
            "bbox" : [ x, y, width, height ],
            "score" : float
            }
        """
        catIds = [self.get_category_id(name) for name in predictions.classnames]
        results = []
        for image_id, img in self.coco.imgs.items():
            det = predictions[str(image_id)]
            width, height = img['width'], img['height']
            scale = np.asarray([1, width, height, width, height])

            for cls, bbox in enumerate(det):
                bbox = bbox * scale  # [conf, cx, cy, w, h]
                bbox[:, 1:3] -= (bbox[:, 3:5] / 2)
                for box in bbox:
                    score = round(box[0], 5)
                    bbox_int = [int(v) for v in box[1:5]]
                    results.append(dict(image_id=image_id, category_id=catIds[cls], bbox=bbox_int, score=score))
        return results

    def evaluate(self, predictions, num_processes=1, metric=None):
        if metric is None:
            metric = self.metric

        if isinstance(metric, Iterable) and not isinstance(metric, str):
            # mulit metric, return last one as major score
            score = dict(score=0)
            for m in self.metric:
                score = self.evaluate(predictions, num_processes=num_processes, metric=m)
            return score

        if metric.lower() == 'voc':
            print_out = metric[0].isupper()
            from .metric import evaluate_detections
            voc_metric = evaluate_detections(predictions, self, num_processes, print_out=print_out)
            print(f'{metric} metric:', voc_metric['mean'])
            return voc_metric
        elif metric == 'COCO':
            # COCO eval
            from pycocotools.cocoeval import COCOeval
            res_anno = list(self.load_prediction(predictions))
            if res_anno:
                res_coco = self.coco.loadRes(res_anno)

                E = COCOeval(self.coco, res_coco, iouType='bbox')  # initialize CocoEval object
                E.evaluate()
                E.accumulate()
                E.summarize()

                return dict(score=E.stats[0])
            else:
                print('WARN no detection at all!')
            return dict(score=0)
        else:
            raise NotImplementedError(metric)

    def resample(self, num_samples=None):
        """resample dataset by weighting classes"""
        # weight by class count
        class_count = category_count(self.coco)
        class_weights = 1. / class_count
        # limit max weight
        max_weight = 1. / len(class_count)
        class_weights[class_weights > max_weight] = max_weight

        # sample weights = sum of class weight each sample
        weights = np.zeros(len(self))
        for index in range(len(self)):
            img_id = self.ids[index]
            target = self._get_target(img_id)
            for obj in target:
                cat_id = obj['category_id']
                weights[index] += class_weights[cat_id]
        num_samples = num_samples or len(self)
        return WeightedRandomDataset(self, weights, num_samples)
