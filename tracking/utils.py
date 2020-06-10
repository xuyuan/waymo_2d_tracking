import json
import numpy as np
from sort.tracker_sort import MultiClassTrackerSort

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

uniq_object_id = 0

IMAGE_SIZES = {
    'FRONT': [1920, 1280],
    'FRONT_LEFT': [1920, 1280],
    'FRONT_RIGHT': [1920, 1280],
    'SIDE_LEFT': [1920, 886],
    'SIDE_RIGHT': [1920, 886]
}


def clip_xy(camera_id, x, y):
    w, h = IMAGE_SIZES[camera_id]
    return np.clip(x, a_min=0, a_max=w), np.clip(y, a_min=0, a_max=h)


def track_sort(predictions, segment_id, camera_id, iou_thresholds, max_age, min_hits):
    camera_predictions = predictions[segment_id][camera_id]

    tracked_predictions = []
    tracker = MultiClassTrackerSort(max_age=max_age, min_hits=min_hits)
    frame_ids = camera_predictions.keys()
    for frame_id in sorted(frame_ids):
        tracker_input = [[
            e['bbox'][0], e['bbox'][1], e['bbox'][0] + e['bbox'][2], e['bbox'][1] + e['bbox'][3],
            e['score'],
            e['category_id']] for e in camera_predictions[frame_id]]
        tracker_result = tracker.track(tracker_input, iou_thresholds)
        for category_id in tracker_result.keys():
            for tracked_object in tracker_result[category_id]:
                # TODO: clip values?
                x1, y1 = tracked_object[0], tracked_object[1]
                x1, y1 = clip_xy(camera_id, x1, y1)
                x2, y2 = tracked_object[2], tracked_object[3]
                x2, y2 = clip_xy(camera_id, x2, y2)
                width, height = x2 - x1, y2 - y1
                if width < 1 or height < 1:
                    continue
                object_id = int(tracked_object[4])
                # TODO: clip values?
                confidence = np.clip(tracked_object[5], a_min=0.2, a_max=1.0)

                image_id = '%s/%i/%s' % (segment_id, frame_id, camera_id)
                tracked_predictions.append({
                    'image_id': image_id,
                    'bbox': [x1, y1, width, height],
                    'score': confidence,
                    'category_id': category_id,
                    'object_id': '%i' % object_id
                })

    return tracked_predictions


def read_data_file(file_name, score_threshold):
    entries = {}
    raw_entries = json.load(open(file_name))
    if 'annotations' in raw_entries:
        raw_entries = raw_entries['annotations']
    for entry in raw_entries:
        image_id = entry['image_id']
        segment_id, frame_id, camera_id = image_id.split('/')
        if segment_id not in entries:
            entries[segment_id] = {}
        if camera_id not in entries[segment_id]:
            entries[segment_id][camera_id] = {}
        frame_id = int(frame_id)
        if frame_id not in entries[segment_id][camera_id]:
            entries[segment_id][camera_id][frame_id] = []
        bbox = entry['bbox']
        if bbox[2] < 1 or bbox[3] < 1:
            # invalid objects
            continue
        category_id = entry['category_id']
        score = 1.0  # assume ground truth
        if 'score' in entry:
            score = entry['score']
        if score < score_threshold[category_id - 1]:
            continue
        new_entry = {
            'bbox': bbox,
            'score': score,
            'category_id': category_id
        }
        if 'object_id' in entry:
            new_entry['object_id'] = entry['object_id']
        entries[segment_id][camera_id][frame_id].append(new_entry)
    return entries
