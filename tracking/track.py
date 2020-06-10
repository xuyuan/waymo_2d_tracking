import json
import time
import argparse
from os.path import dirname, join

from utils import read_data_file, track_sort

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ground-truth", type=str, default='/data/waymo/det2d/validation/images.json',
                        help='ground-truth json')
    parser.add_argument("--input", type=str, default='submission_12545.json',
                        help='submission.json')
    parser.add_argument("--output", type=str, default='tracker_predictions.json',
                        help='file to save the tracker predictions')
    parser.add_argument("--max-age", type=int, default=1, help='SORT max-age')
    parser.add_argument("--min-hits", type=int, default=0, help='SORT min-hits')
    parser.add_argument("--score-threshold", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.95, 0.6, 1.0, 0.9], help='score threshold to track')
    parser.add_argument("--iou-threshold", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.01, 0.01, 1.0, 0.0], help='IOU threshold for tracking')
    parser.add_argument("--segment-id", type=str, help='track only a single segment')
    args = parser.parse_args()
    print(args)

    predictions = read_data_file(args.input, args.score_threshold)
    image_id2path = {}
    ground_truth_dir = dirname(args.ground_truth)
    for image in json.load(open(args.ground_truth)):
        image_id2path[image['id']] = join(ground_truth_dir, image['file_name'])

    tracked_predictions = []

    if args.segment_id:
        predictions = {k: v for k, v in predictions.items() if k in [args.segment_id]}

    start_time = time.time()
    for segment_id in predictions.keys():
        print(segment_id)
        for camera_id in predictions[segment_id]:
            tracked_predictions += track_sort(predictions, segment_id, camera_id,
                                              args.iou_threshold, args.max_age, args.min_hits)

    print("duration: %.2fs" % (time.time() - start_time))
    json.dump(tracked_predictions, open(args.output, 'wt'))
