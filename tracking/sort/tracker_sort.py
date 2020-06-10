#
# Multi class SORT tracker
#

import numpy as np
from .sort import Sort
import warnings


class MultiClassTrackerSort(object):

    def __init__(self, max_age=1, min_hits=0):
        """Efffeu SORT based multi object tracker.

        :param max_age: how many frames should be skipped before the object is removed
        :param min_hits: how many frames must the object detected before it is tracked
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = {}

    def track(self, detected_objects, iou_thresholds):
        """
        Tracks objects from the given detected objects list.
        :param detected_objects: [[x1, y1, x2, y2, confidence, class_name], ...]
        :return: tracked objects {class_name: [[x1, y1, x2, y2, object_id], ...] ... }
        """
        class2detections = {}
        for detected_object in detected_objects:
            bbox_with_confidence = detected_object[:5]
            class_name = detected_object[5]
            if class_name not in self.trackers:
                self.trackers[class_name] = Sort(max_age=self.max_age, min_hits=self.min_hits)
            if class_name not in class2detections:
                class2detections[class_name] = []
            class2detections[class_name].append(bbox_with_confidence)

        all_tracked_objects = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            for class_name in self.trackers:
                class_tracker = self.trackers[class_name]
                if class_name in class2detections:
                    class_detections = class2detections[class_name]
                    all_tracked_objects[class_name] = class_tracker.update(np.array(class_detections, dtype=np.float32),
                                                                           iou_threshold=iou_thresholds[class_name-1])
                else:
                    all_tracked_objects[class_name] = class_tracker.update(np.array([], dtype=np.float32),
                                                                           iou_threshold=iou_thresholds[class_name-1])

        return all_tracked_objects


if __name__ == '__main__':
    tracker = MultiClassTrackerSort(max_age=1, min_hits=0)
    print(tracker.track([[0, 0, 10, 10, 1, 'ship']]))
    print(tracker.track([[0, 0, 10, 10, 1, 'ship']]))
    print(tracker.track([]))
    result = tracker.track([[0, 0, 10, 10, 1, 'ship']])
    print(result)
    print(tracker.track([]))
    #print(tracker.track([]))
    result = tracker.track([[0, 0, 10, 10, 1, 'ship'], [110, 110, 120, 120, 0.5, 'car']])
    print(result)

