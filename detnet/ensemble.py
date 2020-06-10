"""ensemble submission together"""


import argparse
from pathlib import Path
import json
from functools import partial
from collections import defaultdict
import numbers
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml
from tqdm import tqdm
from .nn.tta import merge_detections, nms_detections
from .trainer.utils import get_num_workers



def lxly2cxcy(bbox):
    """[score, left, top, width, height] -> [score, center x, center y, width, height]"""
    bbox[:, 1:3] += bbox[:, 3:5] / 2
    return bbox


def cxcy2lxly(bbox):
    """[score, center x, center y, width, height] -> [score, left, top, width, height]"""
    bbox[:, 1:3] -= bbox[:, 3:5] / 2
    return bbox


def convert_submission(det_list, weight, min_score=0):
    """
    Args:
        det_list: list load from submission json
    Returns:
        {image_id: {category_id: [[score, x, y, w, h], ...], ...}, ...}
    """
    detections = defaultdict(lambda: defaultdict(list))
    for det in det_list:
        image_id = det['image_id']
        bbox = det['bbox']
        if bbox[2] > 0 and bbox[3] > 0:  # filter zero size box
            category_id = det['category_id']
            bbox = [det['score'] * weight] + bbox
            if bbox[0] >= min_score:
                detections[image_id][category_id].append(bbox)
    return detections


def ensemble(image_id, detections, category_ids):
    output_json = []
    for category_id in category_ids:
        bboxes = [det[category_id] for det in detections]
        bboxes = [np.asarray(bbox).reshape(-1, 5) for bbox in bboxes]
        bboxes = [lxly2cxcy(bbox) for bbox in bboxes]
        merged = merge_func(bboxes)
        merged = cxcy2lxly(merged)

        for bbox in merged:
            if bbox[0] > args.min_score:
                output = {'image_id': image_id, 'category_id': category_id,
                          'bbox': bbox[1:].astype(int).tolist(), 'score': round(bbox[0], 5)}
                output_json.append(output)
    return output_json


def load_yml_input_and_weight(input_files_with_weights, prefix=''):
    results = []
    for k, v in input_files_with_weights.items():
        p = prefix + '/' + k if prefix else k
        if isinstance(v, numbers.Number):
            results.append((p, v))
        else:
            results += load_yml_input_and_weight(v, p)
    return results


def load_input_submissions(input_files, input_weights):
    input_submissions = [json.load(Path(f).open()) for f in input_files]

    category_ids = set(sum([[d['category_id'] for d in det] for det in input_submissions], []))
    input_detections = [convert_submission(d, w, args.min_score) for d, w in zip(input_submissions, input_weights)]
    image_ids = set(sum([list(input_detection.keys()) for input_detection in input_detections], []))
    return image_ids, category_ids, input_detections


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('inputs', type=str, nargs='+', help='input json files')
    parser.add_argument('-o', '--output', type=str, help='output json file')
    parser.add_argument('-m', '--method', choices=("weighted_fusion", "nms", "soft_nms"), default="weighted_fusion",
                        help='method to merge bbox detections')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IOU threshold for merging bboxes')
    parser.add_argument('--soft-nms-cut', type=float, default=1.0, help='cutout IoU threshold for soft nms')
    parser.add_argument('--min-score', type=float, default=0, help='minimal score to keep')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='number of workers')
    args = parser.parse_args()

    input_files = []
    for f in args.inputs:
        f = Path(f)
        if f.is_file():
            input_files.append(f)
        elif f.is_dir():
            for f in f.glob("**/*.json"):
                input_files.append(f)
        else:
            print(f"{f} is neither file nor dir?!")

    input_weights = None
    if len(input_files) == 1 and input_files[0].suffix == '.yml':
        input_files_with_weights = yaml.load(input_files[0].open())
        input_files_with_weights = load_yml_input_and_weight(input_files_with_weights)
        input_files, input_weights = zip(*input_files_with_weights)
        print(input_files, input_weights)

    assert len(input_files) > 1
    print('input files:', input_files)

    if not input_weights:
        input_weights = [1] * len(input_weights)
    input_weights_weight = max(input_weights)
    input_weights = [w / input_weights_weight for w in input_weights]
    print('weights', input_weights)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise RuntimeError(f"output file {output_file} exists!")

    image_ids, category_ids, input_detections = load_input_submissions(input_files, input_weights)
    print('No. Images:', len(image_ids))
    print('No. categories:', len(category_ids))

    merge_func = partial(merge_detections, nms_thresh=args.iou_thresh)
    if args.method == 'nms':
        merge_func = partial(nms_detections, iou_thresh=args.iou_thresh)
    elif args.method == 'soft_nms':
        merge_func = partial(nms_detections, iou_thresh=args.iou_thresh, soft=True, soft_nms_cut=args.soft_nms_cut)

    num_workers = get_num_workers(args.jobs)
    output_json = []
    if num_workers == 1:
        for image_id in tqdm(image_ids):
            detections = [input_detection[image_id] for input_detection in input_detections]
            output_json += ensemble(image_id, detections, category_ids)
    else:
        future = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for image_id in tqdm(image_ids):
                detections = [input_detection[image_id] for input_detection in input_detections]
                future.append(ex.submit(ensemble, image_id, detections, category_ids))
        for f in tqdm(future):
            output_json += f.result()

    with output_file.open('wt') as fp:
        json.dump(output_json, fp)

