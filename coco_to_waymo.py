import argparse
import json
from pathlib import Path
from tqdm import tqdm

from waymo_open_dataset import label_pb2, dataset_pb2
from waymo_open_dataset.protos import metrics_pb2, submission_pb2

LABEL_TYPE = {0: label_pb2.Label.TYPE_VEHICLE,
              1: label_pb2.Label.TYPE_PEDESTRIAN,
              2: label_pb2.Label.TYPE_CYCLIST,
              3: label_pb2.Label.TYPE_SIGN}


# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/tools/create_prediction_file_example.py
def create_pd_object(detection, context_name, frame_timestamp_micros, camera_name):
    """Creates a prediction objects file."""
    o = metrics_pb2.Object()
    # The following 3 fields are used to uniquely identify a frame a prediction
    # is predicted at. Make sure you set them to values exactly the same as what
    # we provided in the raw data. Otherwise your prediction is considered as a
    # false negative.
    o.context_name = context_name
    # The frame timestamp for the prediction. See Frame::timestamp_micros in
    # dataset.proto.
    # invalid_ts = -1
    o.frame_timestamp_micros = frame_timestamp_micros
    # This is only needed for 2D detection or tracking tasks.
    # Set it to the camera name the prediction is for.
    o.camera_name = dataset_pb2.CameraName.Name.Value(camera_name)

    bbox, score, label = detection['bbox'], detection['score'], detection['category_id']

    # Populating box and score.
    box = label_pb2.Label.Box()
    box.center_x = bbox[0] + bbox[2] * 0.5
    box.center_y = bbox[1] + bbox[3] * 0.5
    box.length = bbox[2]
    box.width = bbox[3]
    o.object.box.CopyFrom(box)
    # This must be within [0.0, 1.0]. It is better to filter those boxes with
    # small scores to speed up metrics computation.
    o.score = score
    # For tracking, this must be set and it must be unique for each tracked sequence.
    if 'object_id' in detection:
        o.object.id = detection['object_id']
    # Use correct type.
    o.object.type = label
    assert o.object.type != label_pb2.Label.TYPE_UNKNOWN
    return o


def create_pd_objects(detections):
    objects = metrics_pb2.Objects()

    for detection in tqdm(detections):
        # each frame
        context_name, frame_timestamp_micros, camera_name = detection['image_id'].split('/')
        o = create_pd_object(detection, context_name, int(frame_timestamp_micros), camera_name)
        objects.objects.append(o)
    return objects


# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/tools/create_submission.cc
def create_pb_submission(detections, unique_method_name, description, account_name, tracking):
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.TRACKING_2D if tracking else submission_pb2.Submission.DETECTION_2D
    submission.account_name = account_name
    submission.authors.append('Yuan Xu')
    submission.authors.append('Erdene-Ochir Tuguldur')
    submission.affiliation = 'DAInamite'
    submission.unique_method_name = unique_method_name
    submission.description = description
    submission.method_link = ""
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    objects = create_pd_objects(detections)
    submission.inference_results.CopyFrom(objects)
    # submission.object_types = [LABEL_TYPE.items()]  # all types by default
    # submission.latency_second = ?
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detection', type=str, nargs='+', help='detection result json file')
    parser.add_argument('--unique-method-name', type=str, required=True, help='unique method name. Max 25 chars.')
    parser.add_argument('--description', type=str, help='detailed description of method.')
    parser.add_argument('--account-name', type=str, required=True, help='email')
    parser.add_argument('--tracking', action='store_true', help='tracking submission')
    parser.add_argument('-o', '--output', type=str, help='output submission file')
    args = parser.parse_args()

    # load our prediction in pickle
    detections = []
    for f in args.detection:
        detections += json.load(open(f))

    submission = create_pb_submission(detections, args.unique_method_name, args.description,
                                      args.account_name, args.tracking)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(submission.SerializeToString())
