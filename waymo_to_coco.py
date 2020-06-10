"""convert waymo dataset to coco"""

import os

# GPU is not needed anyway
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from pathlib import Path
import cv2
import json

import tensorflow.compat.v1 as tf

from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()
WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']


def get_camera_labels(frame):
    if frame.camera_labels:
        return frame.camera_labels
    return frame.projected_lidar_labels


def extract_segment(segment_path, out_dir, step):
    print(f'extracting {segment_path}')
    segment_name = segment_path.name
    segment_out_dir = out_dir / segment_name
    segment_out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
    for i, data in enumerate(dataset):
        if i % step != 0:
            continue

        print('.', end='', flush=True)
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        context_name = frame.context.name
        frame_timestamp_micros = str(frame.timestamp_micros)

        for index, image in enumerate(frame.images):
            camera_name = open_dataset.CameraName.Name.Name(image.name)
            img = tf.image.decode_jpeg(image.image).numpy()
            image_id = '/'.join([context_name, frame_timestamp_micros, camera_name])
            file_name = image_id + '.jpg'
            filepath = segment_out_dir / file_name
            filepath.parent.mkdir(parents=True, exist_ok=True)

            images.append(dict(file_name=file_name, id=image_id, height=img.shape[0], width=img.shape[1]))
            cv2.imwrite(str(filepath), img)

            for camera_labels in get_camera_labels(frame):
                # Ignore camera labels that do not correspond to this camera.
                if camera_labels.name == image.name:
                    # Iterate over the individual labels.
                    for label in camera_labels.labels:
                        # object bounding box.
                        width = int(label.box.length)
                        height = int(label.box.width)
                        x = int(label.box.center_x - 0.5 * width)
                        y = int(label.box.center_y - 0.5 * height)
                        area = width * height
                        annotations.append(dict(image_id=image_id,
                                                bbox=[x, y, width, height], area=area, category_id=label.type,
                                                object_id=label.id,
                                                tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1))

    with (segment_out_dir / 'annotations.json').open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)


def extract_segment_process(segment_path, out_dir, step):
    os.system(f"python {__file__} --input-segment={segment_path} --out-dir={out_dir} --step={step}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-segment', type=str, required=True, help='path to a Waymo segment TFRecord file')
    parser.add_argument('-o', '--out-dir', type=str, required=True, help='directory to save extracted images')
    parser.add_argument('-t', '--step', type=int, default=10, help='export frame per step')
    parser.add_argument('-j', "--jobs", type=int, default=1, help='Allow N jobs at once')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    segment_path = Path(args.input_segment)
    if segment_path.is_dir():
        tfrecord_files = list(segment_path.glob("*.tfrecord"))
    elif segment_path.suffix == '.tfrecord':
        tfrecord_files = [segment_path]
    else:
        print(f'skip {segment_path}')

    if args.jobs == 1:
        for f in tfrecord_files:
            extract_segment(f, out_dir, args.step)
    else:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs)(delayed(extract_segment_process)(f, out_dir, args.step) for f in tfrecord_files)