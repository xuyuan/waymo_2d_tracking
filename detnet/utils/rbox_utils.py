
# import lazy_import
# lazy_import.lazy_module("cv2")
import cv2
from scipy import optimize
import numpy as np


def generate_mask_from_rboxes(rboxes, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for i, b in enumerate(rboxes):
        x, y, w, h, a = b
        box = cv2.boxPoints(((x, y), (w, h), a))
        box = np.int0(box)
        cv2.drawContours(mask, [box], 0, i + 1, thickness=-1)
    return mask


def generate_rbox(mask):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        biggest_cont = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    else:
        biggest_cont = contours[0]

    points = biggest_cont.reshape(-1, 2)
    (x, y), (w, h), a = cv2.minAreaRect(points)
    x = np.asarray([x, y, w, h, a])

    if w * h > 20:
        mask_area = np.sum(mask)
        def error_fun(x):
            rbox_mask = generate_mask_from_rboxes([x], mask.shape)
            iter = np.sum(rbox_mask * mask)
            union = np.sum(rbox_mask) + mask_area - iter
            iou = iter / union
            if iou > 1:
                import pdb; pdb.set_trace()
            return -iou

        # method       | test on 100 images
        # -------------+-- speed -----+-- F2 -----
        # None         |              | 0.4756
        # Nelder-Mead  | 1.04images/s | 0.4842
        # Powell       | 1.58s/images | 0.4867 0.4906
        # CG           | 1.59images/s | 0.4527
        # BFGS         | 1.60images/s | 0.4527
        opt_res = optimize.minimize(error_fun, x, method='Powell')
        if opt_res.success:
            x = opt_res.x

    return x


def generate_rbox_from_masks(masks, nbox):
    rboxes = []
    for i in range(1, nbox+1):
        mask = (masks == i)
        if np.any(mask):
            rbox = generate_rbox(np.uint8(mask))
            rboxes.append(rbox)
    return np.asarray(rboxes)


def optimize_rbox_mask(masks, nbox=0):
    if nbox == 0:
        nbox = np.max(masks)

    if nbox == 0:
        return masks

    rboxes = generate_rbox_from_masks(masks, nbox)
    rbox_masks = generate_mask_from_rboxes(rboxes, masks.shape)
    return rbox_masks
