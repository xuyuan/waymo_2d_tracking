import random
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageDraw, Image, ImageFont
from skimage.measure import find_contours
from ..trainer.utils import rle_decode
import cv2

LABEL_COLOR = (  # R,   G,   B
    (50, 50, 50),
    (250,  60,  60),
    (000, 200, 200),
    (000, 220, 000),
    (230, 220,  50),
    (240, 000, 130),
    (30,   60, 255),
    (240, 130,  40),
    (160, 000, 200),
    (160, 230,  50),
    (000, 160, 255),
    (230, 175,  45),
    (000, 173, 208),
    (135, 136, 000),
    (185, 219,  14),
    (215, 169, 000),
    (119,  74,  57),
    (156, 162, 153),
    (237,  41,  57),
    (255, 255, 255),
    )


def get_random_color(pastel_factor=0.5):
    return tuple([int((x+pastel_factor)/(1.0+pastel_factor) * 255) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]])


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def get_polygon(mask, color_list):
    contours = find_contours(mask, 0.5)
    color = generate_new_color(color_list, pastel_factor=0.5)
    color_list.append(color)
    p_list = []
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = verts[:, ::-1]
        p = (verts, color)
        p_list.append(p)

    return p_list


def draw_bbox_on_image(image, bboxes, texts=None, relative=False, line_width=None):
    draw = ImageDraw.Draw(image)
    if torch.is_tensor(bboxes):
        bboxes = bboxes.numpy()

    width, height = image.size
    if relative:
        bboxes = np.asarray(bboxes) * [width, height, width, height, 1]

    if line_width is None:
        line_width = int(min(width, height) / 160) + 1
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", line_width * 5)

    for bbox in bboxes:
        label = int(bbox[-1])
        color = LABEL_COLOR[label % len(LABEL_COLOR)]
        draw.rectangle(bbox[:4].tolist(), outline=color, width=line_width)
        if texts:
            draw.text((bbox[0:2] + line_width).tolist(), texts[label], fill=color, font=font)
        else:
            bbox_size = (bbox[2:4] - bbox[:2]).astype(np.int)
            draw.text((bbox[0:2] + line_width).tolist(), f'{bbox_size}', fill=color, font=font)

    return image


def draw_train_sample(sample, classnames=None):
    """
    Args:
        sample (dict):
            input: PIL.Image
            bbox:  [-1, 4] numpy array
            labels: (list of torch.Tensor)
            masks: [C, H, W] numpy array
    """
    img = sample['input'].copy()

    if 'bbox' in sample:
        img = draw_bbox_on_image(img, sample['bbox'], texts=classnames, relative=True)
    if 'masks' in sample:
        img = draw_masks_on_image(img, sample['masks'])
    return img


def draw_ground_truth_on_batch(batch):
    """
    Args:
        batch (dict):
            images: (torch.Tensor) [N, C, H, W] input for training
            bbox: (list of torch.Tensor) [[-1, 4]] * N target for training
            labels: (list of torch.Tensor) [[-1, 1]] * 4
            masks: (torch.LongTensor) [N, 1, H, W]
    """
    if 'image_orig' in batch:
        images = batch['image_orig']
    else:
        images = batch['input']
    boxes = batch.get('bbox', None)
    labels = batch['labels']
    masks = batch.get('masks', None)

    images_drawed = []

    for i in range(len(images)):
        img = to_pil_image(images[i], 'RGB')
        if boxes is not None:
            img = draw_bbox_on_image(img, boxes[i], labels[i])
        if masks is not None:
            img = draw_masks_on_image(img, masks[i])
        img = to_tensor(img)
        images_drawed.append(img)
    return images_drawed


def draw_masks_on_image(image, masks, fill=None):
    """
    :param image: (PIL.Image)
    :param masks: (torch.Tensor)
    :return: (PIL.Image)
    """
    if fill:
        orig_image = image.copy()

    draw = ImageDraw.Draw(image)
    color_list = list()

    scale = np.asarray([image.width / masks.shape[1], image.height / masks.shape[0]])

    for j in range(masks.max()):
        mask_j = masks == (j + 1)
        poly_list = get_polygon(mask_j, color_list)
        for xy, color in poly_list:
            xys = xy * scale
            xys = list(xys.flatten())
            draw.polygon(xys, fill=fill, outline=color)

    if fill:
        image = Image.blend(orig_image, image, 0.5)

    return image


def rle_to_masks(encoded_pixels, shape):
    masks = np.zeros(shape, dtype=np.uint8)
    for i, s in enumerate(encoded_pixels):
        m = rle_decode(s, shape) * (i + 1)
        masks += m
    return masks


def draw_rle_on_image(image, encoded_pixels):
    """
    Parameters
    ----------
    image: PIL image
    encoded_pixels: list of rle encoded masks

    Returns PIL image
    -------
    """
    masks = rle_to_masks(encoded_pixels, (image.height, image.width))
    return draw_masks_on_image(image, masks)


def draw_detection(image, detection, classnames, scale=None, prob_threshold=0, image_threshold=0, seen_classes_number=0):
    if isinstance(detection, tuple):
        bboxes, masks = detection
    else:
        bboxes = detection
        masks = []

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if image_threshold > 0:
        low_max_confidence = True
        for det in bboxes:
            if (det[:, 0] >= image_threshold).any():
                low_max_confidence = False
                break
        if low_max_confidence:
            # none of detection has good confidence
            return image

    if scale is not None:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    scale = np.asarray([image.width, image.height], dtype=np.float32)
    line_width = min(3, int(scale.min() / 640) + 1)
    draw = ImageDraw.Draw(image)
    for i, (name, det) in enumerate(zip(classnames, bboxes)):
        if not all(ord(char) < 128 for char in name):  # only ASCII is possible
            name = 'cls_' + str(i + 1)

        if seen_classes_number > 0:
            color = LABEL_COLOR[int(i > seen_classes_number)]
        else:
            color = LABEL_COLOR[i % len(LABEL_COLOR)]
        for d in det:
            prob, box = d[0], d[1:]
            if prob < prob_threshold:
                continue
            center = box[:2] * scale
            hw = box[2:4] * scale

            if len(box) == 4:  # bbox
                half_wh = hw * 0.5
                points = list(np.asarray([center - half_wh, center + half_wh]).flat)
                draw.rectangle(points, outline=color, width=line_width)
            elif len(box) == 5:  # rbox
                points = cv2.boxPoints((center, hw, box[4]))
                draw.polygon(points, outline=color)
            else:
                raise RuntimeError("wrong size of box: {}".format(len(box)))

            draw.text((center + 1).tolist(), name + " {}".format(int(prob * 100)), fill=color)

    for m in masks:
        image = draw_masks_on_image(image, m)

    return image
