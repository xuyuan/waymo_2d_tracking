"""
export detection results from test.py to submission format
"""
import argparse
from pathlib import Path
import multiprocessing
import csv
import gzip
from functools import lru_cache
from tqdm import tqdm
import warnings
import json
import numpy as np

from .trainer import Predictions
from .utils.visualization import draw_detection, draw_train_sample
from .trainer.data import Subset


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 6)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def export_images(detections, output_dir, dataset, threshold, image_threshold=0, process_id=None):
    classnames = detections.classnames
    warning_msgs = []

    desc = "exporting images"
    if process_id is not None:
        desc += " #{}".format(process_id)
    for sample in tqdm(dataset, desc=desc, position=process_id,
                       mininterval=60, miniters=len(dataset)//100, smoothing=1):
        image_id = sample['image_id']
        det = detections[image_id]
        if det is not None:
            image = draw_train_sample(sample)
            try:
                image = draw_detection(image, det, classnames, prob_threshold=threshold, image_threshold=image_threshold)
            except Exception as e:
                print(det)
                raise e
            fpath = (output_dir / image_id).with_suffix('.png')
            image.save(fpath)
        else:
            msg = f"no detection for image {image_id}"
            if process_id == 0:
                warnings.warn(msg)
            warning_msgs.append(msg)
    return warning_msgs


def export_images_multiprocessing(detections, output_dir, dataset, threshold, num_processes):
    assert (num_processes > 1)
    print('start', num_processes, 'processes')
    pool = multiprocessing.Pool(num_processes)

    dataset_splits = [Subset(dataset, slice(i, None, num_processes)) for i in range(num_processes)]
    worker_args = [
        dict(detections=detections,
             output_dir=output_dir, dataset=subset, threshold=threshold, process_id=i)
        for i, subset in enumerate(dataset_splits)]
    async_ret = [pool.apply_async(export_images, kwds=kwargs) for kwargs in worker_args]
    pool.close()
    pool.join()
    for ret in async_ret:
        for msg in ret.get():
            warnings.warn(msg)


def point_form(center_size):
    """convert bbox (cx, cy, sx, sy) to (x_min, y_min, x_max, y_max)"""
    rect_center = center_size[0:2]
    rect_half_size = center_size[2:4] * 0.5
    return np.asarray([rect_center - rect_half_size, rect_center + rect_half_size]).flatten()


def corner_size(center_size):
    # return (left_top_x, left_top_y, width, height)
    rect_center = center_size[0:2]
    rect_half_size = center_size[2:4] * 0.5
    return np.asarray([rect_center - rect_half_size, center_size[2:4]]).flatten()


def point_form_to_corner_size(xxyy):
    """convert bbox (x_min, y_min, x_max, y_max) to (left_top_x, left_top_y, width, height)"""
    width_height = xxyy[2:4] - xxyy[0:2]
    return np.asarray([xxyy[0], xxyy[1], width_height[0], width_height[1]])


def clip_bbox(bbox, size):
    # size = width, height
    output = np.empty_like(bbox)
    for i in range(2):
        np.clip(bbox[i::2], 0, size[i], out=output[i::2])
    return output


def enumerate_detections(detections, dataset, threshold, image_threshold=0, at_least_one_object=False):
    #classnames = detections.classnames
    classnames = dataset.classnames

    for i, sample in tqdm(enumerate(dataset), disable=None):
        img_id = sample['image_id']
        det = detections[img_id]

        yield i, sample['image_file'], 0, 0, None

        if len(det) == 0:
            # no detections
            continue

        if image_threshold > 0:
            image_gt_threshold = False
            for d in det:
                if (d[:, 0] >= image_threshold).any():
                    image_gt_threshold = True
                    break

            if not image_gt_threshold:
                continue

        thresh = threshold
        if at_least_one_object:
            # at least one object should be detected
            max_conf = [d[:, 0].max() for d in det if len(d) > 0]
            if len(max_conf) == 0:
                continue
            max_conf = max(max_conf)
            thresh = min(max_conf, threshold)

        offset = sample['offset']
        image_size = sample['image_size']
        scale = np.asarray([1, image_size[0], image_size[1], image_size[0], image_size[1]])

        for cls, bbox in zip(classnames[1:], det):
            bbox_conf = bbox[:, 0]
            bbox_filtered = bbox[bbox_conf >= thresh]
            bbox_filtered *= scale
            for box in bbox_filtered:
                rect = point_form(box[1:5])
                rect = clip_bbox(rect, image_size)
                rect = point_form_to_corner_size(rect)
                rect[0:2] += offset
                area = rect[2] * rect[3]
                if area > 1:
                    conf = float(box[0])
                    rect = [int(v) for v in rect]
                    yield i, None, cls, conf, rect


def export_json(detections, output_filename, dataset):
    output_path = Path(output_filename).with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = dataset.load_prediction(detections)
    with open(output_path, 'wt') as fp:
        json.dump(predictions, fp, cls=NumpyArrayEncoder)


def export(detections, output_filename, dataset, format, threshold, num_processes=1):
    parent_dir = Path(output_filename).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        export_json(detections, output_filename, dataset)
    elif format.startswith('Image'):
        output_path = Path(output_filename).parent / "images"
        print(f'exporting images to {output_path}')
        output_path.mkdir(parents=True, exist_ok=True)

        if format == 'Image100':
            dataset = dataset[::100]
        elif format == 'ImageSmallBBox':
            small_bbox = []
            for i, sample in enumerate(dataset):
                bbox = sample['bbox']
                if len(bbox) > 0:
                    bbox_size = (bbox[:, 2] - bbox[:, 0]) - (bbox[:, 3] - bbox[:, 1])
                    if (bbox_size < 32 ** 2).any():
                        small_bbox.append(i)
            dataset = Subset(dataset, small_bbox)

        if num_processes == 1:
            export_images(detections, output_path, dataset, threshold)
        else:
            export_images_multiprocessing(detections, output_path, dataset, threshold, num_processes)


if __name__ == '__main__':
    from data import create_dataset, add_dataset_argument, SlidingWindowDataset

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    add_dataset_argument(parser)

    parser.add_argument('detections', type=str, help='pickle file of detection results')
    parser.add_argument('--format', type=str, choices=('Image', 'Image100', 'ImageSmallBBox', 'json'),
                        default='json', help='the format of input and output')
    parser.add_argument("--threshold", type=float, default=0.05, help='threshold for filtering detection')
    parser.add_argument('-j', '--jobs', type=int, help='number of processes', default=1)
    parser.add_argument('-o', '--output', default='', type=str,
                        help='path for output (default the same as input detections)')
    args = parser.parse_args()

    detections = Predictions.open(args.detections)

    image_size = (args.image_size, args.image_size)
    dataset = create_dataset(args.data_root, mode='test',  data_fold=args.data_fold, image_size=image_size, data_file=args.data_file,
                             use_h5=args.data_h5)

    dataset = SlidingWindowDataset(dataset, np.asarray(image_size))

    if dataset:
        print(dataset)

    output = args.output if args.output else args.detections

    export(detections, output, dataset, args.format, args.threshold, num_processes=args.jobs)
