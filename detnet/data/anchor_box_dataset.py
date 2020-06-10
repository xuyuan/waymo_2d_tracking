
import warnings
from ..trainer.data import Dataset
import torch


class AnchorBoxDataset(Dataset):

    def collate_fn(self, batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """
        image_ids = []
        images = []
        labels = []
        boxes = []
        priors_label = []
        priors_loc = []
        priors_pos = []
        priors_iou = []

        masks = {'heatmap': [],
                 'masks': [],
                 'elevation': []}
        images_orig = []

        for sample in batch:
            try:
                if sample:
                    images.append(sample['input'])
                    if 'priors_label' in sample:
                        priors_label.append(sample['priors_label'])
                        priors_loc.append(sample['priors_loc'])
                        priors_pos.append(sample['priors_pos'])
                    if 'priors_iou' in sample:
                        priors_iou.append(sample['priors_iou'])

                    if 'image_id' in sample:
                        image_ids.append(sample['image_id'])

                    if 'image_orig' in sample:
                        # logging original images
                        images_orig.append(sample['image_orig'])
                    if 'boxes' in sample:
                        boxes.append(torch.from_numpy(sample['boxes']).float())
                        labels.append(torch.from_numpy(sample['labels']))

                    for k, v in masks.items():
                        if k in sample:
                            v.append(torch.from_numpy(sample[k]))
            except RuntimeError as e:
                warnings.warn(str(e), RuntimeWarning)
                return dict()

        if not images:
            return dict()

        ret = dict(input=torch.stack(images))
        if priors_label:
            ret.update(dict(priors_label=torch.stack(priors_label),
                            priors_loc=priors_loc,
                            priors_pos=torch.stack(priors_pos)))
        if priors_iou:
            ret['priors_iou'] = torch.stack(priors_iou)

        for k, v in masks.items():
            if v:
                ret[k] = torch.stack(v)

        if images_orig:
            ret['image_orig'] = torch.stack(images_orig)
        if boxes:
            ret['labels'] = labels
            ret['boxes'] = boxes

        if image_ids:
            ret['image_id'] = image_ids

        return ret
