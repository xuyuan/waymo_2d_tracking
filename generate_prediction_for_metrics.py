import json
import argparse

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", choices=['prediction', 'ground-truth'],
                        default='prediction', help='to generate ground truth or prediction')
    parser.add_argument("--input", type=str, required=True,
                        help='either submission.json or annotations.json')
    parser.add_argument("--output", type=str,
                        default='out.bin', help='output file')
    args = parser.parse_args()
    print(args)

    entries = json.load(open(args.input))
    if args.type == 'ground-truth':
        entries = entries['annotations']

    objects = metrics_pb2.Objects()

    camera_names = {
        'FRONT': dataset_pb2.CameraName.FRONT,
        'FRONT_LEFT': dataset_pb2.CameraName.FRONT_LEFT,
        'FRONT_RIGHT': dataset_pb2.CameraName.FRONT_RIGHT,
        'SIDE_LEFT': dataset_pb2.CameraName.SIDE_LEFT,
        'SIDE_RIGHT': dataset_pb2.CameraName.SIDE_RIGHT
    }
    object_types = {
        1: label_pb2.Label.TYPE_VEHICLE,
        2: label_pb2.Label.TYPE_PEDESTRIAN,
        3: label_pb2.Label.TYPE_SIGN,
        4: label_pb2.Label.TYPE_CYCLIST
    }
    difficulty_levels = {
        1: label_pb2.Label.LEVEL_1,
        2: label_pb2.Label.LEVEL_2
    }

    for e in entries:
        image_id = e['image_id']
        segment_id, frame_id, camera_id = image_id.split('/')

        o = metrics_pb2.Object()
        o.context_name = segment_id
        o.frame_timestamp_micros = int(frame_id)

        o.camera_name = camera_names[camera_id]
        bbox = e['bbox']
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        box = label_pb2.Label.Box()
        box.center_x = center_x
        box.center_y = center_y
        box.center_z = 0
        box.length = bbox[2]
        box.width = bbox[3]
        box.height = 0
        box.heading = 0
        o.object.box.CopyFrom(box)
        o.object.type = object_types[e['category_id']]
        if 'score' in e:
            o.score = e['score']
        if 'object_id' in e:
            o.object.id = e['object_id']
        if 'tracking_difficulty_level' in e:
            o.object.tracking_difficulty_level = difficulty_levels[e['tracking_difficulty_level']]
        if 'detection_difficulty_level' in e:
            o.object.detection_difficulty_level = difficulty_levels[e['detection_difficulty_level']]

        # work around for metrics computation
        o.object.num_lidar_points_in_box = 100

        objects.objects.append(o)

    f = open(args.output, 'wb')
    f.write(objects.SerializeToString())
    f.close()
