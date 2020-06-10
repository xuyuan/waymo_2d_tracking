from pathlib import Path
import numpy as np
from detnet.data import COCODetection
from detnet.trainer.data import TransformedDataset
from detnet.data.metric import evaluate_detections, compute_truth_and_false_positive, voc_ap, f2_score
from detnet.trainer.transforms.vision import ToBGR


def metric_fun(detections, dataset, label_name):
    """Top level function that does the PASCAL VOC evaluation."""
    ovthresh = (0.7,) if label_name == 'vehicle' else (0.5,)
    pos_size, conf, bbox_size, tp_all, fp_all, fn_bbox, tp_mask, fp_mask = compute_truth_and_false_positive(detections, dataset, label_name, ovthresh, ignore_mask=True)

    pos_size = np.concatenate(pos_size)
    npos = len(pos_size)
    # if npos == 0:
    #     print("WARN no sample for {} in dataset".format(label_name))

    sorted_ind = np.argsort(conf)[::-1]
    bbox_size = np.asarray(bbox_size)[sorted_ind]

    results = {}
    for thres in ovthresh:
        k = str(thres)
        tp_s = tp_all[k]
        fp_s = fp_all[k]

        # sort by confidence
        tp_s = np.asarray(tp_s)[sorted_ind]
        fp_s = np.asarray(fp_s)[sorted_ind]

        sizes = {'S': (None, 32**2),
                 'M': (32**2, 96**2),
                 'L': (96**2, None),
                 '': (None, None)}

        for size_key, (low, high) in sizes.items():
            fps = fp_s
            tps = tp_s
            d_size = bbox_size
            p_size = pos_size
            if low is not None:
                mask_low = d_size >= low
                fps = fps[mask_low]
                tps = tps[mask_low]
                d_size = d_size[mask_low]
                p_size = p_size[p_size >= low]
            if high is not None:
                mask_high = d_size < high
                fps = fps[mask_high]
                tps = tps[mask_high]
                d_size = d_size[mask_high]
                p_size = p_size[p_size < high]

            # compute precision recall
            fp = np.cumsum(fps)
            tp = np.cumsum(tps)
            npos = len(p_size)
            rec = tp / np.maximum(npos, np.finfo(np.float64).eps)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec)

            tp_sum = tp[-1] if len(tp) > 0 else 0
            fp_sum = fp[-1] if len(fp) > 0 else 0
            ar = rec[-1] if len(rec) > 0 else float('nan')

            f2 = f2_score(npos, tp_sum, fp_sum)
            results[k+size_key] = {'TP': tp, 'FP': fp, 'FN': fn_bbox[k], 'T': npos,
                                   'recall': rec, 'precision': prec,
                                   'ap': ap, 'f2': f2, 'ar': ar}

    summary = {}
    for k in ['ap', 'ar']:
        # summary[k] = np.mean([v[k] for v in results.values()])
        for t, v in results.items():
            summary[k] = v[k]

    for k in ['T']:
        summary[k] = npos

    summary['score'] = summary['ap']
    return summary


class Dataset(COCODetection):
    def __init__(self, images_root, annotations, ingore_empty=False, bgr=False):
        super(Dataset, self).__init__(images_root, annotations)
        if ingore_empty:
            self.ids = [i for i in self.ids if self._get_target(i)]
        self.to_bgr = ToBGR() if bgr else None

    def getitem(self, index):
        sample = super().getitem(index)

        if 'bbox' in sample:
            # filter bbox
            bbox = sample['bbox']
            valid = (bbox[:, 2] > bbox[:, 0] + 0.01) & (bbox[:, 3] > bbox[:, 1] + 0.01)
            sample['bbox'] = bbox[valid]

        if self.to_bgr:
            sample = self.to_bgr(sample)

        return sample

    def evaluate(self, predictions, num_processes=1):
        evaluation = evaluate_detections(predictions, self, num_processes, metric_fun=metric_fun, print_out=True)

        mean = {}
        for k in ['ap', 'ar', 'f2', 'T']:
            if k in next(iter(evaluation.values())):
                mean[k] = np.mean([v[k] for c, v in evaluation.items() if c != 'sign'])  # ignore sign

        assert mean['T'] > 0 # there is no positive sample in dataset!

        evaluation['mean'] = mean
        evaluation['score'] = mean['ap']

        print(f'metric:', evaluation['mean'])
        return evaluation


def create_dataset(data_root, mode, ingore_empty=False, transform=None, bgr=False):
    data_root = Path(data_root)

    if mode == 'train':
        training_root = data_root / 'training'
        dataset = Dataset(training_root, training_root / 'annotations.json', ingore_empty=ingore_empty, bgr=bgr)
    elif mode != 'all':
        validation_root = data_root / 'validation'
        dataset = Dataset(validation_root, validation_root / 'annotations.json', bgr=bgr)
    else: # 'all'
        dataset = Dataset(data_root, data_root / 'annotations.json', ingore_empty=ingore_empty, bgr=bgr)

    if transform:
        dataset = TransformedDataset(dataset, transform)
    return dataset


def add_dataset_argument(parser):
    group = parser.add_argument_group('options of dataset')
    group.add_argument('--data-root', default='/media/data/waymo/det2d', type=str, help='path to dataset')
    group.add_argument('--data-include-empty', action='store_true', help='include images without objects')
    group.add_argument('--data-bgr', action='store_true', help='images are BGR saved')
    return group