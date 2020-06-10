import warnings
import numpy as np
from ..trainer.data import ImageFolder as ImageFolderBase


class ImageFolder(ImageFolderBase):
    def __init__(self, root, sort_by_image_size=False):
        super().__init__(root, sort_by_image_size=sort_by_image_size)
        self.image_size = {}  # cache image size

    def getitem(self, index):
        sample = super().getitem(index)
        self.image_size[sample['image_id']] = sample['input'].size
        return sample

    def load_prediction(self, predictions):
        """return as COCO annotation format
        """
        if len(self.image_size) != len(self):
            warnings.warn("load all images for image_size")
            for i in range(len(self)):
                self.getitem(i)

        images = []
        annotations = []
        categories = []
        for i, name in enumerate(predictions.classnames):
            categories.append({'id': i+1, 'name': name})

        for image_id, (width, height) in self.image_size.items():
            images.append({'id': image_id, 'file_name': image_id, 'width': width, 'height': height})

            det = predictions[str(image_id)]
            scale = np.asarray([1, width, height, width, height])

            for cls, bbox in enumerate(det):
                bbox = bbox * scale  # [conf, cx, cy, w, h]
                bbox[:, 1:3] -= (bbox[:, 3:5] / 2)  # [conf, x_min, y_min, w, h]
                for box in bbox:
                    score = round(box[0], 5)
                    bbox_int = [int(v) for v in box[1:5]]
                    annotations.append(dict(image_id=image_id, category_id=cls + 1, bbox=bbox_int, score=score))

        # assign id to annotations
        for i, anno in enumerate(annotations):
            anno['id'] = i

        return dict(images=images, annotations=annotations, categories=categories)
