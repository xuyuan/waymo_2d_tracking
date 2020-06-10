import cv2
import json
import random
import argparse
from os.path import dirname, join

from utils import read_data_file, track_sort

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ground-truth", type=str, default='/data/waymo/det2d/validation/images.json',
                        help='ground-truth json')
    parser.add_argument("--input", type=str, default='submission_12611.json', help='submission.json')
    parser.add_argument("--max-age", type=int, default=1, help='SORT max-age')
    parser.add_argument("--min-hits", type=int, default=0, help='SORT min-hits')
    parser.add_argument("--score-threshold", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.95, 0.6, 1.0, 0.9], help='score threshold to track')
    parser.add_argument("--iou-threshold", type=lambda s: [float(item) for item in s.split(',')],
                        default=[0.3, 0.05, 1.0, 0.05], help='IOU threshold for tracking')
    parser.add_argument("--segment-id", type=str, default='10289507859301986274_4200_000_4220_000',
                        help='segment id to track')
    parser.add_argument("--camera-id", type=str, default='FRONT',
                        help='camera id to track')
    parser.add_argument("--replay", action='store_true', help='replay an already tracked submission')
    parser.add_argument("--write-video", action='store_true', help='create a video from visualization')
    args = parser.parse_args()
    print(args)

    if args.replay:
        # don't apply score threshold for replaying
        args.score_threshold = [0 for _ in args.score_threshold]

    predictions = read_data_file(args.input, args.score_threshold)
    image_id2path = {}
    ground_truth_dir = dirname(args.ground_truth)
    for image in json.load(open(args.ground_truth)):
        image_id2path[image['id']] = join(ground_truth_dir, image['file_name'])

    segment_id, camera_id = args.segment_id, args.camera_id

    tracked_predictions_map = {}
    if args.replay:
        tracked_predictions_map = predictions[segment_id][camera_id]
    else:
        tracked_predictions = track_sort(predictions, segment_id, camera_id,
                                         args.iou_threshold, args.max_age, args.min_hits)
        for e in tracked_predictions:
            frame_id = int(e['image_id'].split('/')[1])
            if frame_id not in tracked_predictions_map:
                tracked_predictions_map[frame_id] = []
            tracked_predictions_map[frame_id].append(e)

    video_writer = None
    # visualize
    frame_ids = tracked_predictions_map.keys()
    colors = {}
    for frame_id in sorted(frame_ids):
        image_id = "%s/%i/%s" % (segment_id, frame_id, camera_id)
        img = cv2.imread(image_id2path[image_id])

        for entry in tracked_predictions_map[frame_id]:
            bbox = entry['bbox']
            p1 = int(bbox[0]), int(bbox[1])
            p2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            object_id = entry['object_id']
            if object_id not in colors:
                colors[object_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            score = entry['score']
            cv2.rectangle(img, p1, p2, colors[object_id], 2)

        cv2.imshow('Result', img)
        if args.write_video:
            if video_writer is None:
                h, w = img.shape[:2]
                video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (w, h))
            video_writer.write(img)
        cv2.waitKey(-1)

    if video_writer is not None:
        video_writer.release()

