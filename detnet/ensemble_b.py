"""ensemble submission with ensemble_boxes"""


import argparse
from pathlib import Path
import json
from functools import partial
from collections import defaultdict
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion
from tqdm import tqdm


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 fromfile_prefix_chars='@')
parser.add_argument('inputs', type=str, nargs='+', help='input json files')
parser.add_argument('-o', '--output', type=str, help='output json file')
parser.add_argument('-m', '--method', choices=("weighted_fusion", "nms", "soft_nms", "nmw"), default="weighted_fusion",
                    help='method to merge bbox detections')
parser.add_argument('--iou-thresh', type=float, default=0.5, help='IOU threshold for merging bboxes')
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
assert len(input_files) > 1
print('input files:', input_files)

output_file = Path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)
if output_file.exists():
    raise RuntimeError(f"output file {output_file} exists!")

input_submissions = [json.load(Path(f).open()) for f in input_files]


def ltwh2ltrb(bbox):
    """[left, top, width, height] -> [left, top, right, bottom]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def ltrb2ltwh(bbox):
    """[left, top, right, bottom] -> [left, top, width, height]"""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def convert_submission(det_list):
    """
    Args:
        det_list: list load from submission json
    Returns:
        {image_id: {boxes: [[x1, y1, x2, y2]], scores: [...], labels: [...]}}
    """
    detections = defaultdict(lambda: defaultdict(list))
    for det in det_list:
        image_id = det['image_id']
        bbox = det['bbox']
        if bbox[2] > 0 and bbox[3] > 0:  # filter zero size box
            xyxy = ltwh2ltrb(bbox)
            detections[image_id]['boxes'].append(xyxy)
            detections[image_id]['scores'].append(det['score'])
            detections[image_id]['labels'].append(det['category_id'])
    return detections


category_ids = set(sum([[d['category_id'] for d in det] for det in input_submissions], []))
input_detections = [convert_submission(d) for d in input_submissions]
image_ids = set(sum([list(input_detection.keys()) for input_detection in input_detections], []))
output_json = []
print('No. Images:', len(image_ids))
print('No. categories:', len(category_ids))


if args.method == 'nms':
    merge_func = partial(nms, iou_thr=args.iou_thresh)
elif args.method == 'soft_nms':
    merge_func = partial(soft_nms, sigma=0.1, iou_thr=args.iou_thresh)
elif args.method == 'nmw':
    merge_func = partial(non_maximum_weighted, iou_thr=args.iou_thresh)
elif args.method == 'weighted_fusion':
    merge_func = partial(weighted_boxes_fusion, iou_thr=args.iou_thresh)
else:
    raise NotImplementedError(args.method)

for image_id in tqdm(image_ids):
    boxes = []
    scores = []
    labels = []
    for input_detection in input_detections:
        det = input_detection[image_id]
        if det['labels']:
            boxes.append(det['boxes'])
            scores.append(det['scores'])
            labels.append(det['labels'])

    if labels:
        boxes, scores, labels = merge_func(boxes, scores, labels)
        for box, score, label in zip(boxes, scores, labels):
            output = {'image_id': image_id, 'category_id': int(label),
                      'bbox': ltrb2ltwh(box.tolist()), 'score': round(float(score), 5)}
            output_json.append(output)


with output_file.open('wt') as fp:
    json.dump(output_json, fp)

