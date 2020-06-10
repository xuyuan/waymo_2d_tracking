
import warnings
from io import BytesIO
import collections
import numbers
import random
import copy

import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageDraw
from PIL.ImageFilter import Filter
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as tvt
import skimage
from skimage.exposure import equalize_adapthist
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2
from .common import RandomChoice, RandomChoices, Compose, RandomApply, PassThough


_pil_interpolation_to_str = {
    Image.NEAREST: 'NEAREST',
    Image.BILINEAR: 'BILINEAR',
    Image.BICUBIC: 'BICUBIC',
    Image.LANCZOS: 'LANCZOS',
    None: None
}


def cv2_border_mode_value(border):
    border_value = 0
    if border == 'replicate':
        border_mode = cv2.BORDER_REPLICATE
    elif border == 'reflect':
        border_mode = cv2.BORDER_REFLECT_101
    else:
        border_mode = cv2.BORDER_CONSTANT
        border_value = border
    return dict(borderMode=border_mode, borderValue=border_value)


# normalization for `torchvision.models`
# see http://pytorch.org/docs/master/torchvision/models.html
TORCH_VISION_MEAN = np.asarray([0.485, 0.456, 0.406])
TORCH_VISION_STD = np.asarray([0.229, 0.224, 0.225])
TORCH_VISION_NORMALIZE = tvt.Normalize(mean=TORCH_VISION_MEAN, std=TORCH_VISION_STD)
TORCH_VISION_DENORMALIZE = tvt.Normalize(mean=-TORCH_VISION_MEAN/TORCH_VISION_STD, std=1/TORCH_VISION_STD)


def _random(name, data=(-1, 1)):
    if name == 'choice':
        return np.random.choice(data)
    elif name == 'uniform':
        return np.random.uniform(data[0], data[1])
    else:
        raise NotImplementedError(name)


class VisionTransform(object):
    def __repr__(self): return self.__class__.__name__ + '()'

    def __call__(self, sample):
        """
        :param sample: dict of data, key is used to determine data type, e.g. image, bbox, mask
        :return: transformed sample in dict
        """
        sample = self.pre_transform(sample)
        output_sample = {}
        for k, v in sample.items():
            if k == 'input':
                if isinstance(v, collections.Iterable):
                    output_sample[k] = [self.transform_image(vi) for vi in v]
                elif isinstance(v, Image.Image):
                    output_sample[k] = self.transform_image(v)
                else:
                    raise NotImplementedError(type(v))
            elif k == 'image_h':
                output_sample[k] = self.transform_image_h(v)
            elif k == 'bbox':
                output_sample[k] = self.transform_bbox(v) if len(v) > 0 else v
            elif k.startswith('mask'):
                output_sample[k] = self.transform_mask(v)
            elif k == 'transformed':
                output_sample[k] = sample[k] + [repr(self)]
            else:
                output_sample[k] = sample[k]

        output_sample = self.post_transform(output_sample)
        return output_sample

    def redo(self, sample):
        raise NotImplementedError

    @staticmethod
    def get_input_size(sample):
        img = sample['input']
        if isinstance(img, collections.Iterable):
            img = img[0]
        return img.size

    def pre_transform(self, sample):
        return sample

    def transform_image(self, image):
        return image

    def transform_image_h(self, image):
        raise NotImplementedError

    def transform_bbox(self, bbox):
        return bbox

    def transform_mask(self, mask):
        return mask

    def post_transform(self, sample):
        # if 'input' in sample:
        #     w, h = sample['input'].size
        #     for k, v in sample.items():
        #         if k.startswith('mask'):
        #             if v.shape[0] != h or v.shape[1] != w:
        #                 raise RuntimeError(f'{repr(self)}\n mask size mismatch {(h, w)} != {(v.shape)}')
        return sample

    @staticmethod
    def size_from_number_or_iterable(size, n=2):
        if isinstance(size, numbers.Number):
            return (size,) * n
        elif isinstance(size, collections.Iterable):
            return size
        else:
            raise RuntimeError(type(size))


class Resize(VisionTransform):
    def __init__(self, size, interpolation=None, max_size=None):
        """
        :param size: (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio.
        :param max_size: int: limit the maximum size of longer edge of image, it is useful to avoid OOM
        :param interpolation: interpolation method of PIL, `None` means random
        """
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size

        if self.max_size:
            assert isinstance(size, int)  # max_size only works with not fixed size

    def redo(self, sample):
        return self(sample)

    def __repr__(self):
        arg_str = [f'size={self.size}']
        if self.interpolation:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
            arg_str += [f'interpolation={interpolate_str}']
        if self.max_size:
            arg_str += [f'max_size={self.max_size}']

        arg_str = ', '.join(arg_str)
        return self.__class__.__name__ + '(' + arg_str + ')'

    @staticmethod
    def compute_scaled_image_size(img, size, max_size=None):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return h, w
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            if max_size:
                if oh > max_size:
                    ow = int(max_size / oh * ow)
                    oh = max_size
                if ow > max_size:
                    oh = int(max_size / ow * oh)
                    ow = max_size

            return oh, ow
        return size

    def pre_transform(self, sample):
        self.out_size = Resize.compute_scaled_image_size(sample['input'], self.size, self.max_size)
        self._image_interpolation = self.interpolation if self.interpolation is not None else random.randint(0, 5)
        if 'image_h' in sample:
            self.out_size_h = tuple(int(s1 * s2 / s0) for s0, s1, s2 in zip(sample['input'].size[::-1], sample['image_h'].size[::-1], self.out_size))
        return sample

    def transform_image(self, image):
        return F.resize(image, self.out_size, self._image_interpolation)

    def transform_image_h(self, image):
        interpolation = self.interpolation if self.interpolation is not None else Image.BILINEAR
        return F.resize(image, self.out_size_h, self.interpolation)

    def transform_mask(self, mask):
        return skimage.transform.resize(mask, self.out_size,
                                        order=0, preserve_range=True,
                                        mode='constant', anti_aliasing=False
                                        ).astype(mask.dtype)


class RecordImageSize(VisionTransform):
    def pre_transform(self, sample):
        image = sample['input']
        sample['image_size'] = (image.height, image.width)
        return sample


class Pad(VisionTransform):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1})'.format(self.padding, self.fill)

    def pre_transform(self, sample):
        image = sample['input']
        self.image_size = image.size
        return sample

    def transform_image(self, image):
        if isinstance(self.fill, str):
            return F.pad(image, self.padding, padding_mode=self.fill)
        return F.pad(image, self.padding, fill=self.fill)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            scale_bbox(bbox, self.image_size)
            bbox[:, 0:4:2] += self.padding[0]
            bbox[:, 1:4:2] += self.padding[1]
            new_width = self.image_size[0] + self.padding[0] + self.padding[2]
            new_height = self.image_size[1] + self.padding[1] + self.padding[3]
            normalize_bbox(bbox, (new_width, new_height))
        return bbox

    def transform_mask(self, mask):
        left, top, right, bottom = self.padding
        if isinstance(self.fill, str):
            return np.pad(mask, ((top, bottom), (left, right)), mode=self.fill)
        return np.pad(mask, ((top, bottom), (left, right)), mode='constant', constant_values=0)


class DivisiblePad(Pad):
    def __init__(self, divisible, fill=0):
        super().__init__(0, fill)
        self.divisible = divisible

    def __repr__(self):
        return self.__class__.__name__ + '(divisible={0}, fill={1})'.format(self.divisible, self.fill)

    def pre_transform(self, sample):
        sample = super().pre_transform(sample)
        width, height = self.image_size
        right = self.divisible_padding(width)
        bottom = self.divisible_padding(height)
        self.padding = (0, 0, right, bottom)
        return sample

    def divisible_padding(self, size):
        return int(np.ceil(size / self.divisible) * self.divisible) - size


class RandomPad(Pad):
    def __init__(self, max_padding, fill=0):
        super().__init__(None, fill=fill)
        self.max_padding = self.size_from_number_or_iterable(max_padding)

    def __repr__(self):
        return self.__class__.__name__ + '(max_padding={0}, fill={1})'.format(self.max_padding, self.fill)

    def pre_transform(self, sample):
        sample = super().pre_transform(sample)

        expand_w = int(random.uniform(0, self.max_padding[0]))
        expand_h = int(random.uniform(0, self.max_padding[1]))
        left = int(random.uniform(0, expand_w))
        top = int(random.uniform(0, expand_h))
        right = expand_w - left
        bottom = expand_h - top
        self.padding = (left, top, right, bottom)
        return sample


class PadTo(Pad):
    """Padding image to reach the give size if image is small"""
    def __init__(self, size, fill=0, direction=('bottom', 'right')):
        super().__init__(None, fill=fill)
        self.size = self.size_from_number_or_iterable(size)
        self.direction = direction
        if direction in ('center', 'random'):
            self.direction = (direction, direction)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, fill={1}, direction={2})'.format(self.size, self.fill, self.direction)

    def pre_transform(self, sample):
        sample = super().pre_transform(sample)

        w, h = sample['input'].size
        width = max(0, self.size[1] - w)
        height = max(0, self.size[0] - h)

        height_padding, width_padding = self.direction
        if width_padding == 'left':
            left = width
        elif width_padding == 'right':
            left = 0
        elif width_padding == 'center':
            left = width // 2
        elif width_padding == 'random':
            left = int(random.uniform(0, width))
        else:
            raise NotImplementedError(width_padding)

        if height_padding == 'top':
            top = height
        elif height_padding == 'bottom':
            top = 0
        elif height_padding == 'center':
            top = height // 2
        elif height_padding == 'random':
            top = int(random.uniform(0, height))
        else:
            raise NotImplementedError(height_padding)

        right = width - left
        bottom = height - top
        self.padding = (left, top, right, bottom)
        return sample


class Crop(VisionTransform):
    def __init__(self, size, center=None, truncate_bbox=True, remove_bbox_outside=False):
        """
        :param size: output size
        :param center: center position of output in original image, `None` for center of original image, `str` for random methods
        :param truncate_bbox: truncate bbox to output size
        :param remove_bbox_outside: remove bbox which center is outside of output
        """
        self.size = self.size_from_number_or_iterable(size)

        if center is None or isinstance(center, str):
            self.center = center
        else:
            self.center = self.size_from_number_or_iterable(center)

        self.truncate_bbox = truncate_bbox
        self.remove_bbox_outside = remove_bbox_outside
        self._redoing = False

    def pre_transform(self, sample):
        if self._redoing:
            self._redoing = False
            return sample

        image = sample['input']
        width, height = image.size
        h, w = self.size
        hl, wl = h // 2, w // 2

        if self.center is None:
            x, y = width // 2, height // 2
        elif isinstance(self.center, str):
            x = width // 2 + int(_random(self.center) * (width // 2 - wl))
            y = height // 2 + int(_random(self.center) * (height // 2 - hl))
        else:
            x, y = self.center
            if x < 0:
                x = width + x
            if y < 0:
                y = height + y

        y1 = np.clip(y - hl, 0, height)
        y2 = np.clip(y - hl + h, 0, height)
        x1 = np.clip(x - wl, 0, width)
        x2 = np.clip(x - wl + w, 0, width)

        self.crop_rectangle = (x1, y1, x2, y2)
        self.image_size = image.size
        return sample

    def redo(self, sample):
        self._redoing = True
        return self(sample)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, center={1}, truncate_bbox={2}, remove_bbox_outside={3})'.format(
            self.size, self.center, self.truncate_bbox, self.remove_bbox_outside)

    def transform_image(self, image):
        return image.crop(self.crop_rectangle)

    def transform_image_h(self, image):
        lw, lh = self.image_size
        hw, hh = image.size
        ws = hw / lw
        hs = hh / lh

        x1, y1, x2, y2 = self.crop_rectangle
        crop_rectangle = (int(ws * x1), int(hs * y1), int(ws * x2), int(hs * y2))
        return image.crop(crop_rectangle)

    @staticmethod
    def crop_bbox(bbox, crop_rectangle, image_size, truncate_bbox, remove_bbox_outside):
        x1, y1, x2, y2 = crop_rectangle
        w, h = image_size
        # scale
        w_s = w / (x2 - x1)
        h_s = h / (y2 - y1)

        x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
        if remove_bbox_outside:
            # keep overlap with gt box IF center in sampled patch
            centers = (bbox[:, 0:2] + bbox[:, 2:4]) / 2.0

            # mask in all gt boxes that above and to the left of centers
            m1 = (x1 < centers[:, 0]) * (y1 < centers[:, 1])

            # mask in all gt boxes that under and to the right of centers
            m2 = (x2 > centers[:, 0]) * (y2 > centers[:, 1])
        else:
            m1 = (x1 < (bbox[:, 2])) * (y1 < (bbox[:, 3]))
            m2 = (x2 > (bbox[:, 0])) * (y2 > (bbox[:, 1]))

        # mask in that both m1 and m2 are true
        mask = m1 * m2
        # take only matching gt boxes
        bbox = bbox[mask, :].copy()

        # adjust to crop (by subtracting crop's left,top)
        bbox[:, :4] -= (x1, y1, x1, y1)
        # scale
        bbox[:, :4] *= (w_s, h_s, w_s, h_s)

        # translate boxes
        if truncate_bbox:
            bbox[:, 0:2] = np.maximum(bbox[:, 0:2], 0)
            bbox[:, 2:4] = np.minimum(bbox[:, 2:4], 1)
        return bbox

    def transform_bbox(self, bbox):
        return Crop.crop_bbox(bbox, self.crop_rectangle, self.image_size, self.truncate_bbox, self.remove_bbox_outside)

    def transform_mask(self, mask):
        x1, y1, x2, y2 = self.crop_rectangle
        return mask[y1:y2, x1:x2]


class RandomCrop(Crop):
    def __init__(self, min_size, max_size, max_aspect_ratio=2, truncate_bbox=True, remove_bbox_outside=False, focus=False):
        super().__init__(min_size, truncate_bbox=truncate_bbox, remove_bbox_outside=remove_bbox_outside)
        self.min_size = self.size_from_number_or_iterable(min_size)
        self.max_size = self.size_from_number_or_iterable(max_size)

        assert self.min_size[0] <= self.max_size[0]
        assert self.min_size[1] <= self.max_size[1]

        self.max_aspect_ratio = max_aspect_ratio
        self.focus = focus

    def __repr__(self):
        return self.__class__.__name__ + '(min_size={0}, max_size={1}, max_aspect_ratio={2}, truncate_bbox={3}, remove_bbox_outside={4}, focus={5})'.format(self.min_size, self.max_size, self.max_aspect_ratio, self.truncate_bbox, self.remove_bbox_outside, self.focus)

    def pre_transform(self, sample):
        image = sample['input']
        width, height = image.size

        max_width = min(width, self.max_size[1])
        max_height = min(height, self.max_size[0])

        w = np.random.uniform(self.min_size[1], max_width)
        h = np.random.uniform(self.min_size[0], max_height)

        # aspect ratio constraint
        if w / h > self.max_aspect_ratio:
            w = int(h * self.max_aspect_ratio)
        elif h / w > self.max_aspect_ratio:
            h = int(w * self.max_aspect_ratio)

        left = np.random.uniform(0, width - w)
        top = np.random.uniform(0, height - h)

        if self.focus:
            if 'bbox' in sample:
                bbox = sample['bbox']
                if len(bbox) > 0:
                    box = random.choice(bbox)
                    left = np.random.uniform(box[2] * width - w, box[0] * width)
                    top = np.random.uniform(box[3] * height - h, box[1] * height)
                    left = max(0, min(left, width - w))
                    top = max(0, min(top, height - h))

        self.size = (int(h), int(w))
        self.center = (int(left + w // 2), int(top + h // 2))
        return super().pre_transform(sample)


class CutOut(VisionTransform):
    def __init__(self, max_size, fill=0):
        """
        Randomly mask out one patches from an image.
        :param max_size:
        :param fill: value to fill, or `None` for noise
        """
        self.max_size = self.size_from_number_or_iterable(max_size)
        self.fill = fill

    def __repr__(self):
        return self.__class__.__name__ + f'(max_size={self.max_size}, fill={self.fill})'

    def pre_transform(self, sample):
        image = sample['input']
        width, height = image.size
        y = np.random.randint(height)
        x = np.random.randint(width)
        h = int(np.random.uniform(1, self.max_size[0] / 2))
        w = int(np.random.uniform(1, self.max_size[1] / 2))
        y1 = np.clip(y - h, 0, height)
        y2 = np.clip(y + h, 0, height)
        x1 = np.clip(x - w, 0, width)
        x2 = np.clip(x + w, 0, width)
        self.crop_rectangle = (x1, y1, x2, y2)
        self.image_size = image.size
        return sample

    def transform_image(self, image):
        image = image.copy()
        if self.fill is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle(self.crop_rectangle, fill=self.fill)
        else:
            # past random noise
            pos = (self.crop_rectangle[0], self.crop_rectangle[1])
            w = self.crop_rectangle[2] - self.crop_rectangle[0]
            h = self.crop_rectangle[3] - self.crop_rectangle[1]
            if len(image.getbands()) == 1:
                patch = Image.fromarray(np.uint8(np.random.rand(h, w) * 255))
            else:
                patch = Image.fromarray(np.uint8(np.random.rand(h, w, len(image.getbands())) * 255))
            image.paste(patch, pos)
        return image

    def transform_image_h(self, image):
        return image

    @staticmethod
    def cutout_bbox(bbox, crop_rectangle, image_size):
        x1, y1, x2, y2 = crop_rectangle
        w, h = image_size
        m1 = (x1 / w < (bbox[:, 0])) * (y1 / h < (bbox[:, 1]))
        m2 = (x2 / w > (bbox[:, 2])) * (y2 / h > (bbox[:, 3]))

        # mask that boxes are totally covered
        mask = m1 * m2

        bbox = np.delete(bbox, np.argwhere(mask), axis=0)
        return bbox

    def transform_bbox(self, bbox):
        return CutOut.cutout_bbox(bbox, self.crop_rectangle, self.image_size)

    def transform_mask(self, mask):
        mask = mask.copy()
        x1, y1, x2, y2 = self.crop_rectangle
        mask[y1:y2, x1:x2] = 0
        return mask


class RandomCopyPaste(VisionTransform):
    """Randomly copy and paste one patches from an image.
    """
    def __init__(self, min_size, max_size, copy_dataset=None):
        self.min_size = min_size
        self.max_size = max_size
        self.copy_dataset = copy_dataset

    def __repr__(self):
        return self.__class__.__name__ + f'(min_size={self.min_size}, max_size={self.max_size})'

    def pre_transform(self, sample):
        center = np.random.random(2)
        size = np.random.uniform(self.min_size / 2, self.max_size / 2, 2)
        p0 = np.clip(center - size, 0, 1)
        p1 = np.clip(center + size, 0, 1)
        self.copy_rectangle = (p0, p1)
        self.copy_sample = sample
        if self.copy_dataset:
            self.copy_sample = random.choice(self.copy_dataset)
        self.paste_pos = np.maximum(0, np.random.random(2) - size * 2)
        self.image_size = sample['input'].size
        return sample

    def _copy(self, image):
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, np.ndarray):
            h, w = image.shape
        else:
            raise NotImplementedError(type(image))
        left = int(self.copy_rectangle[0][0] * w)
        right = int(self.copy_rectangle[1][0] * w)
        upper = int(self.copy_rectangle[0][1] * h)
        lower = int(self.copy_rectangle[1][1] * h)
        box = (left, upper, right, lower)
        if isinstance(image, Image.Image):
            return image.crop(box)
        else:
            return image[upper:lower, left:right]

    def _paste(self, patch, dst_image):
        if isinstance(dst_image, Image.Image):
            width, height = dst_image.size
        elif isinstance(dst_image, np.ndarray):
            height, width = dst_image.shape
        else:
            raise NotImplementedError(type(dst_image))

        pos = int(self.paste_pos[0] * width), int(self.paste_pos[1] * height)
        if isinstance(dst_image, Image.Image):
            dst_image.paste(patch, pos)
        else:
            lower = min(height, pos[1] + patch.shape[0])
            right = min(width, pos[0] + patch.shape[1])
            dst_image[pos[1]:lower, pos[0]:right] = patch[:lower-pos[1], :right-pos[0]]

    def transform_image(self, image):
        src_image = self.copy_sample['input']
        patch = self._copy(src_image)
        self._paste(patch, image)
        return image

    def transform_image_h(self, image):
        src_image = self.copy_sample['image_h']
        patch = self._copy(src_image)
        self._paste(patch, image)
        return image

    def transform_bbox(self, bbox):
        src_bbox = self.copy_sample['bbox']
        src_w, src_h = self.copy_sample['input'].size
        left = int(self.copy_rectangle[0][0] * src_w)
        right = int(self.copy_rectangle[1][0] * src_w)
        upper = int(self.copy_rectangle[0][1] * src_h)
        lower = int(self.copy_rectangle[1][1] * src_h)
        crop_rectangle = (left, upper, right, lower)
        crop_width = right - left
        crop_height = lower - upper

        paste_left = int(self.paste_pos[0] * self.image_size[0])
        paste_upper = int(self.paste_pos[1] * self.image_size[1])
        paste_right = paste_left + crop_width
        paste_lower = paste_upper + crop_height
        paste_rectangle = (paste_left, paste_upper, paste_right, paste_lower)
        bbox = CutOut.cutout_bbox(bbox, paste_rectangle, self.image_size)

        patch = Crop.crop_bbox(src_bbox, crop_rectangle, (src_w, src_h), True, True)
        if len(patch) > 0:
            scale_bbox(patch, (crop_width, crop_height))
            patch[:, 0:4:2] += paste_left
            patch[:, 1:4:2] += paste_upper
            normalize_bbox(patch, self.image_size)
            bbox = np.vstack((bbox, patch))
        return bbox

    def transform_mask(self, mask):
        # TODO: here assume only one mask 'masks' in sample
        src_mask = self.copy_sample['masks']
        patch = self._copy(src_mask)
        self._paste(patch, mask)
        return mask


class MaskedTransform(VisionTransform):
    def __init__(self, transform, mask=None):
        """

        Args:
            transform:
            mask: mask array, or callable(mask), or None for random triangle
        """
        self.transform = transform
        self.mask = mask

    def __repr__(self):
        return self.__class__.__name__ + f'(transform={self.transform}, mask={self.mask})'

    def pre_transform(self, sample):
        self._new_sample = self.transform(sample)

        if self.mask is None:
            width, height = self._new_sample['input'].size
            mask = np.zeros((height, width))
            points = np.asarray([(np.random.randint(width), np.random.randint(height)) for i in range(4)])
            cv2.fillConvexPoly(mask, points, 1)
        elif callable(self.mask):
            width, height = self._new_sample['input'].size
            mask = np.zeros((height, width))
            mask = self.mask(mask)
        else:
            mask = self.mask
        self._mask = mask
        return sample

    def transform_image(self, image):
        new_img = self._new_sample['input']
        inv_mask = Image.fromarray(((1 - self._mask) * 255).astype(np.uint8))
        new_img.paste(image, inv_mask)
        return new_img


class GridMask(VisionTransform):
    """
    ref:
        - http://arxiv.org/abs/2001.04086
        - https://www.kaggle.com/haqishen/gridmask

    Args:
        num_grid (int, or (int, int)): number of grid in a row or column.
        fill_value (int, float, list of int, list of float): value for dropped pixels.
        rotate ((float, float) or float): range from which a random angle is picked.
                If rotate is a single int an angle is picked from (-rotate, rotate).
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)
    """
    def __init__(self, num_grid=3, fill=0, rotate=0, mode=0):

        self.num_grid = self.size_from_number_or_iterable(num_grid)
        self.fill = fill
        if isinstance(rotate, numbers.Number):
            rotate = (-rotate, rotate)
        self.rotate = rotate
        self.mode = mode
        self._masks = None
        self._rand_h_max = []
        self._rand_w_max = []
        self._masks_width = 0
        self._masks_height = 0

    def __repr__(self):
        return self.__class__.__name__ + f'(num_grid={self.num_grid}, fill={self.fill}, rotate={self.rotate}, mode={self.mode})'

    def init_masks(self, height, width):
        if self._masks is None or self._masks_height != height or self._masks_width != width:
            self._masks = []
            self._rand_h_max = []
            self._rand_w_max = []
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                        int(i * grid_h): int(i * grid_h + grid_h / 2),
                        int(j * grid_w): int(j * grid_w + grid_w / 2)
                        ] = 0
                        if self.mode == 2:
                            this_mask[
                            int(i * grid_h + grid_h / 2): int(i * grid_h + grid_h),
                            int(j * grid_w + grid_w / 2): int(j * grid_w + grid_w)
                            ] = 0

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self._masks.append(this_mask)
                self._rand_h_max.append(grid_h)
                self._rand_w_max.append(grid_w)

    def pre_transform(self, sample):
        width, height = VisionTransform.get_input_size(sample)
        self.init_masks(height, width)

        mid = np.random.randint(len(self._masks))
        mask = self._masks[mid]
        rand_h = np.random.randint(self._rand_h_max[mid])
        rand_w = np.random.randint(self._rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[0] != 0 or self.rotate[1] != 0 else 0

        self._param = {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}
        return sample

    def transform_image(self, image):
        mask = self._param['mask']

        angle = self._param['angle']
        if angle != 0:
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.rotate(angle)
            mask = np.asarray(mask)

        w, h = image.size
        rand_h, rand_w = self._param['rand_h'], self._param['rand_w']
        mask = mask[rand_h:rand_h + h, rand_w:rand_w + w]
        return _mask_image(image, mask, self.fill)


def _mask_image(image, mask, fill_value):
    image = np.asarray(image)
    if image.ndim == 3:
        mask = mask[:, :, np.newaxis]

    if fill_value is None:  # noise
        fill_value = np.uint8(np.random.rand(*image.shape) * 255)
    elif isinstance(fill_value, collections.Iterable):
        mask = np.repeat(mask, 3, axis=2)
        fill_value = np.uint8(fill_value)

    image = image * mask + fill_value * (1 - mask)
    return Image.fromarray(image)


class FMask(VisionTransform):
    def __init__(self, ratio=0.5, fill=0, decay=3):
        self.ratio = ratio
        self.fill = fill
        self.decay = decay

    def __repr__(self):
        return self.__class__.__name__ + f'(ratio={self.ratio}, fill={self.fill}, decay={self.decay})'

    def pre_transform(self, sample):
        width, height = VisionTransform.get_input_size(sample)
        self._mask, ratio = f_mask([height, width], self.ratio, self.decay)
        return sample

    def transform_image(self, image):
        mask = self._mask[0].astype(np.uint8)
        return _mask_image(image, mask, self.fill)


class MixUp(VisionTransform):
    """
    Suppressing Model Overfitting for Image Super-Resolution Networks
    """
    def __init__(self, dataset, alpha=1.2):
        self.alpha = alpha
        self.dataset = dataset

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, dataset={self.dataset})'

    def pre_transform(self, sample):
        self.copy_sample = random.choice(self.dataset)
        self.weight = np.random.beta(self.alpha, self.alpha)
        return sample

    @staticmethod
    def blend(im1, im2, weight):
        if isinstance(im1, Image.Image):
            # rotate if images have different width
            if im1.width != im2.width:
                im2 = im2.transpose(Image.TRANSPOSE)
            return Image.blend(im1, im2, weight)
        else:
            # CHW image, rotate if images have different width
            if im1.shape[-1] != im2.shape[-1]:
                im2 = im2.transpose(-1, -2)
            return im1 * (1.0 - weight) + im2 * weight

    def transform_image(self, image):
        src_image = self.copy_sample['input']
        return MixUp.blend(image, src_image, self.weight)

    def transform_image_h(self, image):
        src_image = self.copy_sample['image_h']
        return MixUp.blend(image, src_image, self.weight)

    def transform_bbox(self, bbox):
        raise NotImplementedError

    def transform_mask(self, mask):
        raise NotImplementedError


class _ImageHighResSync(VisionTransform):
    """the high resolution image has to be transformed the same way as low resolution image"""
    def transform_image(self, image):
        raise NotImplementedError

    def transform_image_h(self, image):
        return self.transform_image(image)

    def redo(self, sample):
        return self(sample)


class _ImageNoise(VisionTransform):
    """the high resolution image should not be affected"""
    def transform_image(self, image):
        raise NotImplementedError

    def transform_image_h(self, image):
        return image


class HorizontalFlip(_ImageHighResSync):
    def transform_image(self, image):
        return F.hflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 0:4:2] = 1 - bbox[:, -3::-2]
        return bbox

    def transform_mask(self, mask):
        return np.fliplr(mask)


class VerticalFlip(_ImageHighResSync):
    def transform_image(self, image):
        return F.vflip(image)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox[:, 1:4:2] = 1 - bbox[:, -2::-2]
        return bbox

    def transform_mask(self, mask):
        return np.flipud(mask)


class Transpose(_ImageHighResSync):
    def transform_image(self, image):
        return image.transpose(Image.TRANSPOSE)

    def transform_bbox(self, bbox):
        bbox = bbox.copy()
        if bbox.size:
            bbox = bbox[:, [1, 0, 3, 2, 4]]
        return bbox

    def transform_mask(self, mask):
        return mask.T


class ToRGB(_ImageHighResSync):
    def transform_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


class ToYUV(_ImageHighResSync):
    def transform_image(self, image):
        if image.mode != 'YCbCr':
            image = image.convert('YCbCr')
        return image


class ToBGR(_ImageHighResSync):
    def transform_image(self, image):
        a = np.asarray(image)
        a = a[...,::-1]
        return Image.fromarray(a)


class Grayscale(_ImageHighResSync):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def transform_image(self, image):
        return F.to_grayscale(image, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class ColorJitter(_ImageHighResSync, tvt.ColorJitter):
    _redoing = False

    def pre_transform(self, sample):
        if self._redoing:
            self._redoing = False
            return sample

        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)
        return sample

    def transform_image(self, image):
        return self.transform(image)

    def __repr__(self):
        return tvt.ColorJitter.__repr__(self)

    def redo(self, sample):
        self._redoing = True
        return self(sample)


class Posterize(_ImageNoise):
    def __init__(self, bits):
        """
        Reduce the number of bits for each color channel.
        :param bits: The number of bits to keep for each channel (1-8).
        """
        self.bits = bits

    def __repr__(self):
        return self.__class__.__name__ + '(bits={0})'.format(self.bits)

    def transform_image(self, image):
        if self.bits > 0:
            return ImageOps.posterize(image, self.bits)
        return image


class Solarize(_ImageHighResSync):
    def __init__(self, threshold=128):
        """
        Invert all pixel values above a threshold.
        :param threshold: All pixels above this greyscale level are inverted.
        """
        self.threshold = threshold

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0})'.format(self.threshold)

    def transform_image(self, image):
        return ImageOps.solarize(image, self.threshold)


class SolarizeAdd(_ImageHighResSync):
    def __init__(self, addition, threshold=128):
        """add 'addition' amount to pixel if it is less than threshold"""
        self.threshold = threshold
        self.addition = addition

        lut = np.arange(256)
        lut_add = lut < threshold
        lut[lut_add] = np.clip(lut[lut_add] + addition, 0, 255)
        self.lut = np.tile(lut, 3)

    def transform_image(self, image):
        # lut = self.lut
        #if image.mode == "L":
        #    lut = lut[:256]
        return image.point(self.lut)


class AutoContrast(VisionTransform):
    def transform_image(self, image):
        try:
            return ImageOps.autocontrast(image)
        except IOError as e:
            warnings.warn(str(e))
            return image


class Equalize(VisionTransform):
    def transform_image(self, image):
        return ImageOps.equalize(image)


class Invert(_ImageHighResSync):
    def transform_image(self, image):
        return ImageOps.invert(image)


def clahe(image):
    with warnings.catch_warnings():
        # ignore skimage/util/dtype.py:135: UserWarning: Possible precision loss when converting from float64 to uint16
        warnings.simplefilter("ignore")
        return Image.fromarray((equalize_adapthist(np.asarray(image)) * 255).astype(np.uint8))


class CLAHE(VisionTransform):
    """Contrast Limited Adaptive Histogram Equalization"""
    def transform_image(self, image):
        return clahe(image)


class SaltAndPepper(_ImageNoise):
    def __init__(self, probability=0.001):
        self.probability = probability

    def __repr__(self):
        return self.__class__.__name__ + f'(probability={self.probability})'

    def transform_image(self, image):
        w, h = image.size
        noise = np.random.rand(h, w)
        probability = np.random.uniform(0, self.probability)
        threshold = 1 - probability

        salt = np.uint8(noise > threshold) * 255
        salt = Image.fromarray(salt)
        salt = salt.convert(image.mode)
        image = ImageChops.lighter(image, salt)

        pepper = np.uint8(noise > probability) * 255
        pepper = Image.fromarray(pepper)
        pepper = pepper.convert(image.mode)
        image = ImageChops.darker(image, pepper)

        return image

    def redo(self, sample):
        return self(sample)


class GaussNoise(_ImageNoise):
    def __init__(self, sigma=0.1, per_channel=False):
        self.sigma = sigma
        self.per_channel = per_channel

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma}, per_channel={self.per_channel})'

    def transform_image(self, image):
        image = np.asarray(image)

        shape = list(image.shape)
        if not self.per_channel:
            shape[-1] = 1

        gauss = np.random.normal(0, self.sigma ** 2, shape) * 255
        noisy = image + gauss.astype(np.uint8)
        return Image.fromarray(noisy)


class ImageTransform(VisionTransform):
    enhance_ops = None

    def __init__(self, factor):
        self.factor = factor

    def __repr__(self):
        return self.__class__.__name__ + '(factor={0})'.format(self.factor)

    def transform_image(self, image):
        return self.enhance_ops(image).enhance(1 + self.factor)


class RandomImageTransform(ImageTransform):
    def __init__(self, factors=1.0, random='uniform'):
        super().__init__(1)
        if isinstance(factors, numbers.Number):
            assert factors >= 0
            factors = (-factors, factors)
        self.factors = factors
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + '(factors={0}, random={1})'.format(self.factors, self.random)

    def pre_transform(self, sample):
        self.factor = _random(self.random, self.factors)
        return super().pre_transform(sample)


class EnhanceColor(ImageTransform):
    enhance_ops = ImageEnhance.Color


class RandomAdjustColor(RandomImageTransform):
    enhance_ops = ImageEnhance.Color


class EnhanceContrast(ImageTransform):
    enhance_ops = ImageEnhance.Contrast


class RandomContrast(RandomImageTransform):
    enhance_ops = ImageEnhance.Contrast


class RandomSharpness(RandomImageTransform):
    enhance_ops = ImageEnhance.Sharpness

    def transform_image_h(self, image):
        return image


class RandomBrightness(RandomImageTransform):
    enhance_ops = ImageEnhance.Brightness


class JpegCompression(_ImageNoise):
    def __init__(self, min_quality=45, max_quality=95):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __repr__(self):
        return self.__class__.__name__ + f'(min_quality={self.min_quality}, max_quality={self.max_quality})'

    def transform_image(self, image):
        bytes_io = BytesIO()
        quality = int(np.random.uniform(self.min_quality, self.max_quality))
        image.save(bytes_io, 'JPEG', quality=quality)
        bytes_io.seek(0)
        return Image.open(bytes_io)


class ImageFilter(_ImageNoise):
    def __init__(self, filter):
        self.filter = filter

    def __repr__(self):
        return self.__class__.__name__ + f'(filter={self.filter})'

    def transform_image(self, image):
        return image.filter(self.filter)


class ToByteTensor(_ImageHighResSync):
    def transform_image(self, image):
        a = np.asarray(image)
        t = torch.from_numpy(a)
        return t


class ToTensor(_ImageHighResSync):
    def __init__(self, scaling=True, normalize=TORCH_VISION_NORMALIZE):
        """
        convert image to torch.tensor
        Args:
            scaling: if tensors are scaled in the range [0.0, 1.0]
            normalize: `torchvision.transforms.Normalize` or None
        """
        self.scaling = scaling
        self.normalize = normalize

    def transform_image(self, image):
        # 3D image in format CDHW, e.g. CT scans
        image_3d = isinstance(image, np.ndarray) and (image.ndim in {4})

        if image_3d:
            # CDHW --> CD(H*W)
            image_3d_shape = image.shape
            image = image.reshape((image_3d_shape[0], image_3d_shape[1], -1))
            image_t = torch.from_numpy(image)
        elif isinstance(image, (list, tuple)):
            images = [ToTensor.to_tensor(i, self.scaling) for i in image]
            image_t = torch.stack(images)
            image_t = image_t.permute((1, 0, 2, 3))  # DCHW --> CDHW
            # CDHW --> CD(H*W)
            image_3d = True
            image_3d_shape = image_t.shape
            image_t = image_t.view((image_3d_shape[0], image_3d_shape[1], -1))
        else:
            image_t = ToTensor.to_tensor(image, self.scaling)

        if self.normalize:
            image_t = self.normalize(image_t)

        if image_3d:
            image_t = image_t.view(image_3d_shape)

        return image_t

    def transform_mask(self, mask):
        # numpy to tensor
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(mask)

    @staticmethod
    def to_tensor(pil_img, scaling):
        if scaling:
            return F.to_tensor(pil_img)
        else:
            np_img = np.float32(pil_img)
            return torch.as_tensor(np_img.transpose(2, 0, 1))

    def __repr__(self):
        return self.__class__.__name__ + f'(scaling={self.scaling}, normalize={self.normalize})'


class ToPILImage(_ImageHighResSync):
    """undo `ToTensor`"""
    def __init__(self, denormalize=TORCH_VISION_DENORMALIZE, scaling=True):
        self.denormalize = denormalize
        self.scaling = scaling

    def pre_transform(self, sample):
        sample = sample.copy()
        sample['input'] = sample.pop('input')
        return sample

    def transform_image(self, image):
        if self.denormalize:
            image = self.denormalize(image)
        if not self.scaling:
            image /= 255
        image = torch.clamp(image, 0, 1)
        return F.to_pil_image(image)

    def transform_mask(self, mask):
        return mask.numpy()

    def __repr__(self):
        return self.__class__.__name__ + f'(denormalize={self.denormalize}, scaling={self.scaling})'


def scale_bbox(boxes, image_size):
    """
    :param boxes: [:, 4] float array in range [0, 1)
    :param image_size: [width, height]
    :return: [:, 4] float array of pixel coordinates
    """
    if boxes.size:
        boxes[:, 0:4:2] *= image_size[0]
        boxes[:, 1:4:2] *= image_size[1]
    return boxes


class ScaleBBox(VisionTransform):
    def __call__(self, sample):
        image = sample['input']
        boxes = sample['bbox']
        scale_bbox(boxes, image.size)
        return sample


def normalize_bbox(boxes, image_size):
    """
    :param boxes: [:, 4] float array of pixel coordinates
    :param image_size: [width, height]
    :return: [:, 4] float array in range [0, 1)
    """
    if boxes.size:
        boxes[:, 0:4:2] /= image_size[0]
        boxes[:, 1:4:2] /= image_size[1]
    return boxes


class NormalizeBBox(VisionTransform):
    def __call__(self, sample):
        image = sample['input']
        boxes = sample['bbox']
        normalize_bbox(boxes, image.size)
        return sample


class PerspectiveTransform(_ImageHighResSync):
    def __init__(self, dst_corners, border='replicate'):
        self.dst_corners = dst_corners
        self.border = border

    def __repr__(self):
        return self.__class__.__name__ + f'(dst_corners={self.dst_corners}, border={self.border})'

    @staticmethod
    def get_perspective_transform(dst_corners, width, height):
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        box1 = np.asarray(dst_corners, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        return cv2.getPerspectiveTransform(box0, box1)

    @staticmethod
    def warp_perspective(image_array, mat, border, interpolation):

        height, width = image_array.shape[:2]

        return cv2.warpPerspective(image_array, mat, (width, height), flags=interpolation,
                                   **cv2_border_mode_value(border))

    def pre_transform(self, sample):
        image = sample['input']
        if isinstance(image, collections.Iterable):
            image = image[0]
        self.perspective_matrix = self.get_perspective_transform(self.dst_corners,
                                                                 image.width, image.height)
        self.image_size = image.size
        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = self.warp_perspective(image_array, self.perspective_matrix, self.border, cv2.INTER_LINEAR)
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        labels = bbox[:, -1:]
        bbox = bbox[:, (0, 1, 2, 3, 0, 3, 2, 1)]  # N, 8
        src = bbox.reshape(-1, 2) * self.image_size
        dst = cv2.perspectiveTransform(src.reshape(1, -1, 2), self.perspective_matrix)
        bbox = (dst / self.image_size).reshape(-1, 4, 2)
        bbox_min = bbox.min(1)
        bbox_max = bbox.max(1)
        bbox = np.hstack((bbox_min, bbox_max))
        return np.hstack((bbox, labels))

    def transform_mask(self, mask):
        border = self.border if self.border == 'replicate' else None
        return self.warp_perspective(mask, self.perspective_matrix, border, cv2.INTER_NEAREST)


class Translate(PerspectiveTransform):
    def __init__(self, trans, border='replicate'):
        super().__init__(None, border)
        self.trans = trans

    def __repr__(self):
        return self.__class__.__name__ + f'(trans={self.trans}, border={self.border})'

    @property
    def trans(self):
        return self._trans

    @trans.setter
    def trans(self, t):
        assert isinstance(t, collections.Iterable)
        assert len(t) == 2
        self._trans = t
        self.dst_corners = [t, [1 + t[0], t[1]], [1 + t[0], 1 + t[1]], [t[0], 1 + t[1]]]


class RandomTranslate(Translate):
    def __init__(self, max_trans, border='replicate', random='uniform'):
        super().__init__((0, 0), border)
        if isinstance(max_trans, numbers.Number):
            assert max_trans > 0
            max_trans = (max_trans, max_trans)
        self.max_trans = max_trans
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_trans={self.max_trans}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.trans = tuple(_random(self.random) * t for t in self.max_trans)
        return super().pre_transform(sample)

    def redo(self, sample):
        raise NotImplementedError


class HorizontalShear(PerspectiveTransform):
    def __init__(self, shear, border='replicate'):
        super().__init__(None, border)
        self.shear = shear

    def __repr__(self):
        return self.__class__.__name__ + f'(shear={self.shear}, border={self.border})'

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, dx):
        self._shear = dx
        self.dst_corners = [[dx, 0], [1 + dx, 0], [1 - dx, 1], [-dx, 1]]


class RandomHorizontalShear(HorizontalShear):
    def __init__(self, max_shear, border='replicate', random='uniform'):
        super().__init__(0, border)
        if isinstance(max_shear, numbers.Number):
            assert max_shear > 0
            max_shear = (-max_shear, max_shear)
        self.max_shear = max_shear
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_shear={self.max_shear}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.shear = _random(self.random, self.max_shear)
        return super().pre_transform(sample)

    def redo(self, sample):
        raise NotImplementedError


class VerticalShear(PerspectiveTransform):
    def __init__(self, shear, border='replicate'):
        super().__init__(None, border)
        self.shear = shear

    def __repr__(self):
        return self.__class__.__name__ + f'(shear={self.shear}, border={self.border})'

    @property
    def shear(self):
        return self._shear

    @shear.setter
    def shear(self, dy):
        self._shear = dy
        self.dst_corners = [[0, dy], [1, -dy], [1, 1 - dy], [0, 1 + dy]]


class RandomVerticalShear(VerticalShear):
    def __init__(self, max_shear, border='replicate', random='uniform'):
        super().__init__(0, border)
        if isinstance(max_shear, numbers.Number):
            assert max_shear > 0
            max_shear = (-max_shear, max_shear)
        self.max_shear = max_shear
        self.random = random

    def __repr__(self):
        return self.__class__.__name__ + f'(max_shear={self.max_shear}, border={self.border}, random={self.random})'

    def pre_transform(self, sample):
        self.shear = _random(self.random, self.max_shear)
        return super().pre_transform(sample)

    def redo(self, sample):
        raise NotImplementedError


class Skew(PerspectiveTransform):
    #        TopLeft, BottomLeft, BottomRight, TopRight
    direction_table = np.asarray([[[-1, 0], [0, 0], [0, 0], [0, 0]],
                                  [[0, -1], [0, 0], [0, 0], [0, 0]],
                                  [[0, -1], [0, 0], [0, 0], [0, +1]],
                                  [[0, 0], [0, 0], [0, 0], [0, +1]],
                                  [[0, 0], [0, 0], [0, 0], [-1, 0]],
                                  [[0, 0], [0, 0], [+1, 0], [-1, 0]],
                                  [[0, 0], [0, 0], [+1, 0], [0, 0]],
                                  [[0, 0], [0, 0], [0, +1], [0, 0]],
                                  [[0, 0], [0, -1], [0, +1], [0, 0]],
                                  [[0, 0], [0, -1], [0, 0], [0, 0]],
                                  [[0, 0], [+1, 0], [0, 0], [0, 0]],
                                  [[-1, 0], [+1, 0], [0, 0], [0, 0]]])

    def __init__(self, magnitude, direction=0, border='replicate'):
        """
        perspective skewing on images in one of 12 different direction.
        :param magnitude: percentage of image's size
        :param direction: skew direction as clock, e.g. 12 is skewing up, 6 is skewing down, 0 means random
        """
        super().__init__(None, border)
        self.magnitude = magnitude
        self.direction = direction

    def __repr__(self):
        return self.__class__.__name__ + f'(magnitude={self.magnitude}, direction={self.direction}, border={self.border})'

    def pre_transform(self, sample):
        direction = self.direction - 1 if self.direction != 0 else np.random.randint(0, len(self.direction_table))
        corners = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.dst_corners = corners + (self.direction_table[direction] * self.magnitude)
        return super().pre_transform(sample)


class RandomSkew(Skew):
    def __init__(self, max_magnitude, direction=0, border='replicate'):
        super().__init__(0, direction, border)
        self.max_magnitude = max_magnitude

    def __repr__(self):
        return self.__class__.__name__ + f'(max_magnitude={self.max_magnitude}, direction={self.direction}, border={self.border})'

    def pre_transform(self, sample):
        self.magnitude = np.random.uniform(-self.max_magnitude, self.max_magnitude)
        return super().pre_transform(sample)

    def redo(self, sample):
        raise NotImplementedError


class GridDistortion(VisionTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, axis=None, border='replicate'):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.axis = axis
        self.border = border

    def __repr__(self):
        return self.__class__.__name__ + f'(num_steps={self.num_steps}, distort_limit={self.distort_limit}, axis={self.axis}, border={self.border})'

    def get_rand_steps(self, width):
        xsteps = 1 + np.random.uniform(-self.distort_limit, self.distort_limit, self.num_steps + 1)
        x_step = width // self.num_steps

        xx = np.zeros(width, np.float32)
        prev = 0
        for idx, x in zip(range(len(xsteps)), range(0, width, x_step)):
            start = x
            end = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * xsteps[idx]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur
        return xx

    def pre_transform(self, sample):
        width, height = sample['input'].size

        if self.axis is None or self.axis == 1:
            steps_x = self.get_rand_steps(width)
        else:
            steps_x = np.arange(width, dtype=np.float32)
        if self.axis is None or self.axis == 0:
            steps_y = self.get_rand_steps(height)
        else:
            steps_y = np.arange(height, dtype=np.float32)
        self.mesh_grid = np.meshgrid(steps_x, steps_y)
        self.steps_x = steps_x
        self.steps_y = steps_y

        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = cv2.remap(image_array, self.mesh_grid[0], self.mesh_grid[1],
                                interpolation=cv2.INTER_LINEAR, **cv2_border_mode_value(self.border))
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        h, w = self.mesh_grid[0].shape
        scale_bbox(bbox, (w, h))
        bbox[:, 0] = np.searchsorted(self.steps_x, bbox[:, 0])
        bbox[:, 1] = np.searchsorted(self.steps_y, bbox[:, 1])
        bbox[:, 2] = np.searchsorted(self.steps_x, bbox[:, 2])
        bbox[:, 3] = np.searchsorted(self.steps_y, bbox[:, 3])

        # clip area
        width_height = bbox[:, 2:4] - bbox[:, :2]
        area = width_height[:, 0] * width_height[:, 1]
        bbox = bbox[area > 1]

        normalize_bbox(bbox, (w, h))
        return bbox

    def transform_mask(self, mask):
        return cv2.remap(mask, self.mesh_grid[0], self.mesh_grid[1],
                         interpolation=cv2.INTER_NEAREST, **cv2_border_mode_value(self.border))


class ElasticDeformation(VisionTransform):
    """Elastic deformation."""
    def __init__(self, alpha=1000, sigma=30, approximate=False):
        self.alpha = alpha
        self.sigma = sigma
        self.approximate = approximate

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma}, approximate={self.approximate})'

    @staticmethod
    def elastic_indices(shape, alpha, sigma, approximate):
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

        :see also:
            - https://github.com/albu/albumentations/blob/master/albumentations/augmentations/functional.py#L461
        """
        dx = (np.random.rand(*shape).astype(np.float32) * 2 - 1)
        dy = (np.random.rand(*shape).astype(np.float32) * 2 - 1)

        if approximate:
            # Approximate computation smooth displacement map with a large enough kernel.
            # On large images (512+) this is approximately 2X times faster
            ksize = sigma * 4 + 1
            cv2.GaussianBlur(dx, (ksize, ksize), sigma, dst=dx)
            cv2.GaussianBlur(dy, (ksize, ksize), sigma, dst=dy)
        else:
            dx = gaussian_filter(dx, sigma, mode="constant", cval=0)
            dy = gaussian_filter(dy, sigma, mode="constant", cval=0)

        dx *= alpha
        dy *= alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.asarray([x + dx,
                              y + dy])
        return indices

    @staticmethod
    def elastic_transform(image, indices, spline_order, mode='nearest'):
        if len(image.shape) == 3:
            result = np.empty_like(image)
            for i in range(image.shape[-1]):
                map_coordinates(image[..., i], indices, output=result[..., i], order=spline_order, mode=mode)
            return result
        return map_coordinates(image, indices, order=spline_order, mode=mode)

    def pre_transform(self, sample):
        image = sample['input']
        shape = (image.height, image.width)
        self.indices = self.elastic_indices(shape, self.alpha, self.sigma, self.approximate)
        return super().pre_transform(sample)

    def transform_image(self, image):
        image_array = np.asarray(image)
        image_array = self.elastic_transform(image_array, self.indices, 1)
        return Image.fromarray(image_array)

    def transform_bbox(self, bbox):
        raise NotImplementedError()

    def transform_mask(self, mask):
        return self.elastic_transform(mask, self.indices, 0)


class Rotate(_ImageHighResSync):
    def __init__(self, angle, interpolation=None, expand=False, fill=None):
        """
        :param angle: Rotation angle in degrees in counter-clockwise direction.
        :param expand: If true, expands the output to make it large enough to hold the entire rotated image.
        :param fill: color for area outside the rotated image.
        """
        self.angle = angle
        self.interpolation = interpolation
        self.expand = expand
        self.fill = fill

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + f'(angle={self.angle}, interpolation={interpolate_str}, expand={self.expand}, fill={self.fill})'

    def pre_transform(self, sample):
        image = sample['input']
        if isinstance(image, collections.Iterable):
            image = image[0]
        self.image_size = image.size
        return super().pre_transform(sample)

    def transform_image(self, image):
        interpolation = self.interpolation if self.interpolation is not None else random.choice((Image.NEAREST, Image.BILINEAR, Image.BICUBIC))
        try:
            image = image.rotate(angle=self.angle, resample=interpolation, expand=self.expand, fillcolor=self.fill)
        except TypeError:
            image = image.rotate(angle=self.angle, resample=interpolation, expand=self.expand)  # for PIL version 5
        return image

    def transform_bbox(self, bbox):
        """
        Rotates the bbox coordinated by degrees.
        """
        if self.expand:
            raise NotImplementedError('rotate bbox does not support expanded image')

        radians = np.deg2rad(self.angle)
        labels = bbox[:, -1:]
        bbox = bbox[:, :-1]

        # step 1. Translate the bbox to the center of the image
        bbox -= 0.5

        # step 2. turn the normalized 0-1 coordinates to absolute pixel locations.
        scale_bbox(bbox, self.image_size)

        # step 3. 2 points --> 4 points
        bbox = bbox[:, (0, 1, 2, 3, 0, 3, 2, 1)]

        # step 4. view as points
        points = bbox.reshape(-1, 2)

        # step 5. flip y
        points[:, 1] *= -1

        # Rotate the coordinates in counter-clockwise direction
        rotation_matrix = np.stack(
            [[np.cos(radians), np.sin(radians)],
             [-np.sin(radians), np.cos(radians)]])
        new_points = np.dot(points, rotation_matrix)

        # step 5. flip y
        new_points[:, 1] *= -1

        # step 4. view as Nx4x2
        new_bbox = new_points.reshape(-1, 4, 2)

        # step 3. min / max value as bounding box
        min_xy = new_bbox.min(axis=1)
        max_xy = new_bbox.max(axis=1)
        new_bbox = np.hstack((min_xy, max_xy))

        # step 2.  convert them back to normalized 0-1 floats
        normalize_bbox(new_bbox, self.image_size)

        # step 1. translate center
        new_bbox += 0.5

        # Clip the bboxes to be sure the fall between [0, 1].
        np.clip(new_bbox, 0, 1, out=new_bbox)

        # clip area
        width_height = new_bbox[:, 2:] - new_bbox[:, :2]
        area = width_height[:, 0] * width_height[:, 1] * (self.image_size[0] * self.image_size[1])
        new_bbox = np.hstack((new_bbox, labels))
        return new_bbox[area > 1]

    def transform_mask(self, mask):
        return skimage.transform.rotate(mask, self.angle, mode='constant', resize=self.expand,
                                        preserve_range=True, order=0).astype(mask.dtype)


class RandmonRotate(Rotate):
    def __init__(self, degrees, interpolation=None, expand=False, fill=None, random='uniform'):
        super().__init__(0, interpolation=interpolation, expand=expand, fill=fill)
        self.random = random
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + f'(degrees={self.degrees}, interpolation={interpolate_str}, expand={self.expand}, fill={self.fill}, random={self.random})'

    def pre_transform(self, sample):
        self.angle = _random(self.random, self.degrees)
        return super().pre_transform(sample)

    def redo(self, sample):
        raise NotImplementedError


class HighRes2LowRes(_ImageNoise):
    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(interpolation={0})'.format(interpolate_str)

    def pre_transform(self, sample):
        self.image_h = sample['image_h']
        return sample

    def transform_image(self, image):
        w, h = image.size
        return F.resize(self.image_h, (h, w), self.interpolation)


class TryApply(object):
    def __init__(self, transform: VisionTransform, max_trail=1, fallback_origin=True):
        self.transform = transform
        self.max_trail = max_trail
        self.fallback_origin = fallback_origin

    def __repr__(self):
        return self.__class__.__name__ + f'(transform={self.transform}, max_trail={self.max_trail}, fallback_origin={self.fallback_origin})'

    def __call__(self, sample):
        n_bbox = len(sample.get('bbox', []))
        n_mask = {k: np.any(v) for k, v in sample.items() if k.startswith('mask')}

        for trail in range(self.max_trail):
            sample_copy = copy.deepcopy(sample)
            sample_t = self.transform(sample_copy)

            # valid transform doens't remove all bbox or all mask
            n_bbox_n = len(sample.get('bbox', []))
            n_mask_n = {k: np.any(v) for k, v in sample.items() if k.startswith('mask')}
            assert n_bbox == n_bbox_n
            for k in n_mask:
                assert n_mask[k] == n_mask_n[k]

            bbox_ok = (n_bbox == 0 or len(sample_t.get('bbox', [])) > 0)
            if bbox_ok:
                mask_ok = True
                for k, v in n_mask.items():
                    if v and (not np.any(sample_t[k])):
                        mask_ok = False
                        break

                if mask_ok:
                    return sample_t
        if self.fallback_origin:
            return sample # fall back to no changes
        else:
            return sample_t


class ApplyMultiBBoxAugmentation(VisionTransform):
    def __init__(self, transform, prob):
        """Applies aug_func to the image for each bbox in bboxes.
        """
        super().__init__()
        self.transform = transform
        self.prob = prob

    def __repr__(self):
        return self.__class__.__name__ + f'(transform={self.transform}, prob={self.prob})'

    def __call__(self, sample):
        bboxes = sample.get('bbox', [])
        if len(bboxes) == 0:
            return sample

        image = sample['input']
        scale_bbox(bboxes, image.size)
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2
        box_sizes = np.abs(bboxes[:, 2:4] - bboxes[:, :2])[:, ::-1]  # H, W

        for c, s, b in zip(centers, box_sizes, bboxes):
            if (s < 1).any():
                # ignore too small bbox
                continue

            if self.prob < random.random():
                continue
            crop_func = Crop(size=s, center=c)
            box_sample = crop_func(dict(input=image))

            if box_sample['input'].width == 0 or box_sample['input'].height == 0:
                # warnings.warn(f"cropped image size 0! {image.size} -- {s} {c} --> {box_sample['input'].size}")
                continue

            box_sample = self.transform(box_sample)
            box_image = box_sample['input']

            image.paste(box_image, (int(b[0]), int(b[1])))

        normalize_bbox(bboxes, image.size)
        return sample


def f_mask(size, ratio, decay_power=3):
    mask = make_low_freq_image(decay_power, size)
    mask, ratio = binarise_mask(mask, ratio, size)
    return mask, ratio


def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def binarise_mask(mask, weight, in_shape):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param weight: Mean value of final mask
    :param in_shape: Shape of inputs
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = int(weight * mask.size)
    weight = num / mask.size  # adjust weight to exactly match pixel ratio

    mask[idx[:num]] = 1
    mask[idx[num:]] = 0

    mask = mask.reshape((1, *in_shape))
    return mask, weight


class AugMix(RandomChoice, VisionTransform):
    """AugMix:  A Simple Data Processing Method to Improve Robustness and Uncertainty
    https://github.com/google-research/augmix"""
    def __init__(self, transforms, width=3, depth=-1, alpha=1.):
        """
        Perform AugMix augmentations and compute mixture.
        Args:
           severity: Severity of underlying augmentation operators (between 1 to 10).
           width: Width of augmentation chain
           depth: Depth of augmentation chain. -1 enables stochastic depth uniformly from [1, 3]
           alpha: Probability coefficient for Beta and Dirichlet distributions.
         Returns:
           mixed: Augmented and mixed image.
        """
        RandomChoice.__init__(self, transforms)
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def pre_transform(self, sample):
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width)) * (1 - m)

        self._sample_augs = [(m, sample)]
        for i in range(self.width):
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            augs = RandomChoices(self.transforms, depth)
            sample_aug = augs(sample)
            self._sample_augs.append((ws[i], sample_aug))

        return sample

    def transform_image(self, image):
        images = sum([np.asarray(sample['input']) * w for w, sample in self._sample_augs])
        return Image.fromarray(images)