import numpy as np
from ..trainer.utils import rle_decode, get_num_workers
import pickle
from torch import multiprocessing
import pandas as pd
import torch


def filter_detections(det, label, threshold):
    if isinstance(det, tuple):
        det_i = tuple(vv[label] for vv in det)
        if threshold > 0 and det_i[0].size > 0:
            conf = det_i[0][:, 0] > threshold
            bbox = det_i[0][conf]
            masks = [m for j, m in enumerate(det[1]) if conf[j]]
            det_i = (bbox, masks)
    else:
        det_i = det[label]
        if threshold > 0 and det_i.size > 0:
            try:
                conf = det_i[:, 0] > threshold
                det_i = det_i[conf]
            except Exception as e:
                print('det', det, label)
                raise e
    return det_i


def voc_eval(detections, dataset, label_name, ovthresh=(0.5,), size_ovthreshs=0.5, use_07_metric=False):
    """Top level function that does the PASCAL VOC evaluation."""
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

        sizes = {'': (None, None)}
        if size_ovthreshs == thres:
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
            ap = voc_ap(rec, prec, use_07_metric=use_07_metric)

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
            summary[k + "@{}".format(t)] = v[k]

    for k in ['T']:
        summary[k] = npos

    summary['score'] = summary['ap@0.5']
    return summary


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_truth_and_false_positive(detections, dataset, label_name, ovthresh, ignore_mask):
    # N = number of sorted detections over all
    conf = []
    bbox_size = []

    # N = number of images
    pos_size = []

    # {threshold: N number of sorted detections}
    tp_bbox = {}
    fp_bbox = {}

    # {threshold: list of bbox which are not detected at all}
    fn_bbox = {}

    tp_mask = {}
    fp_mask = {}

    for thres in ovthresh:
        k = str(thres)
        tp_bbox[k] = []
        fp_bbox[k] = []
        tp_mask[k] = []
        fp_mask[k] = []
        fn_bbox[k] = []

    label = dataset.classnames.index(label_name)

    for sample in dataset:
        # ground truth
        image_id = sample['image_id']
        image = sample['input']
        image_size = image.shape[-2:] if torch.is_tensor(image) else image.size
        image_area = image_size[0] * image_size[1]
        bbox = sample['bbox']
        bbox_label = bbox[:, -1].astype(int)
        label_mask = bbox_label == label
        bbox_t = bbox[label_mask]
        if 'difficulties' in sample:
            difficulties = sample['difficulties'][label_mask]
        else:
            difficulties = np.zeros(len(bbox_t))
        nt = len(bbox_t)
        masks_t = sample.get('instance_masks', None)
        bbox_t_size = (bbox_t[:, 2] - bbox_t[:, 0]) * (bbox_t[:, 3] - bbox_t[:, 1])
        pos_size.append(bbox_t_size * image_area)

        # detection
        det = detections.get(image_id)
        if det is None:
            continue

        if isinstance(det, tuple):
            bboxes, masks_d = det[0], det[1]
        else:
            bboxes = det
            masks_d = None

        if bboxes.size == 0:
            continue

        confidence = np.array(bboxes[:, 0])
        bbox_d = np.array(bboxes[:, 1:])

        # sort by confidence, make sure we match by order of confidence
        sorted_ind = np.argsort(confidence)[::-1]
        confidence = confidence[sorted_ind]
        conf += confidence.tolist()

        no_mask = ignore_mask or masks_d is None or masks_t is None

        if nt == 0:
            for bbox in bbox_d:
                area = (bbox[2] * bbox[3])
                bbox_size.append(area * image_area)
            for thres in ovthresh:
                # FP
                k = str(thres)
                tp_bbox[k] += [0] * len(bboxes)
                fp_bbox[k] += [1] * len(bboxes)
                if not no_mask:
                    tp_mask[k] += [0] * len(masks_d)
                    fp_mask[k] += [1] * len(masks_d)
            continue

        # go down dets and mark TPs and FPs
        bbox_t_matched_all = {str(k): np.zeros(nt) for k in ovthresh}

        for bbox in bbox_d:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbox_t[:, 0], bbox[0] - bbox[2] / 2)
            iymin = np.maximum(bbox_t[:, 1], bbox[1] - bbox[3] / 2)
            ixmax = np.minimum(bbox_t[:, 2], bbox[0] + bbox[2] / 2)
            iymax = np.minimum(bbox_t[:, 3], bbox[1] + bbox[3] / 2)
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            area = (bbox[2] * bbox[3])
            bbox_size.append(area * image_area)
            inters = iw * ih
            uni = (area + bbox_t_size - inters)
            overlaps = inters / uni
            update_tp_fp(overlaps, difficulties, ovthresh, tp_bbox, fp_bbox, bbox_t_matched_all)

        # print FN
        for thresh, bbox_t_matched in bbox_t_matched_all.items():
            fn = bbox_t_matched != 1
            if fn.any():
                for bbox in bbox_t[fn].tolist():
                    fn_bbox[thresh].append(bbox)

        if no_mask:
            continue

        masks_t_matched_all = {str(k): np.zeros(nt) for k in ovthresh}
        masks_t_area = np.asarray([np.sum(masks_t == i+1) for i in range(nt)])  # instance id for one class
        for mask in masks_d:
            m = rle_decode(mask, masks_t.shape)
            inters = masks_t * m
            inters_area = np.asarray([np.sum(inters == i+1) for i in range(nt)])
            uni_area = masks_t_area + np.sum(m) - inters_area
            overlaps = inters_area / uni_area
            update_tp_fp(overlaps, difficulties, ovthresh, tp_mask, fp_mask, masks_t_matched_all)

    for thresh in fn_bbox:
        fn_bbox[thresh] = np.asarray(fn_bbox[thresh])

    return pos_size, conf, bbox_size, tp_bbox, fp_bbox, fn_bbox, tp_mask, fp_mask



def update_tp_fp(overlaps, difficulties, ovthresh, tp_bbox, fp_bbox, bbox_t_matched_all):
    jmax = np.argmax(overlaps)
    ovmax = overlaps[jmax]
    difficulty = difficulties[jmax]

    for thres in ovthresh:
        k = str(thres)
        tp = tp_bbox[k]
        fp = fp_bbox[k]
        bbox_t_matched = bbox_t_matched_all[k]
        if ovmax > thres:
            if difficulty:
                # ignore difficulties
                tp.append(0)
                fp.append(0)
            elif not bbox_t_matched[jmax]:
                # TP
                tp.append(1)
                fp.append(0)
                bbox_t_matched[jmax] = 1
            else:
                # FP
                tp.append(0)
                fp.append(1)
        else:
            # FP
            tp.append(0)
            fp.append(1)


def f2_score(npos, tp, fp, beta=2):
    if not isinstance(npos, np.ndarray) and npos == 0:
        f2 = 1 if fp == 0 else 0
    else:
        fn = npos - tp
        f2 = (1 + beta ** 2) * tp
        f2_div = ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)

        if isinstance(npos, np.ndarray):
            null_pos = (npos == 0)
            if np.any(null_pos):
                f2[null_pos] = (fp[null_pos] == 0).astype(np.int)
                f2_div[null_pos] = 1

        f2 = f2 / f2_div
    return f2


def evaluate_detections(detections, dataset, num_processes=1, print_out=False, metric_fun=voc_eval, threshold=0.01, output=None):
    evaluation = {}
    classnames = dataset.classnames

    # compatible with old model has background in classnames
    if classnames[0] == 'background':
        classnames = classnames[1:]


    num_processes = get_num_workers(num_processes)
    num_processes = min(num_processes, len(classnames))
    if print_out:
        if dataset:
            print('Evaluating on {} images for {} classes in {} process'.format(len(dataset), len(classnames), num_processes))
        else:
            print('Evaluating on {} classes in {} process'.format(len(classnames), num_processes))

    if num_processes > 1:
        pool = multiprocessing.Pool(num_processes)

    # metric per class
    for i, cls in enumerate(classnames):
        if dataset and cls not in dataset.classnames:
            # print('WARN no label for {} in dataset'.format(cls))
            continue

        detection_per_class = {}
        for k, v in detections:
            detection_per_class[k] = filter_detections(v, i, threshold)
        if num_processes == 1:
            rec_prec_ap = metric_fun(detection_per_class, dataset, cls)
            evaluation[cls] = rec_prec_ap
        else:
            evaluation[cls] = pool.apply_async(metric_fun, (detection_per_class, dataset, cls))

    if num_processes > 1:
        pool.close()
        pool.join()
        for cls, async_ret in evaluation.items():
            rec_prec_ap = async_ret.get()
            evaluation[cls] = rec_prec_ap

    if print_out:
        df = pd.DataFrame.from_dict(evaluation)
        if metric_fun == voc_eval:
            df = df.transpose()
            # df = df[['T', 'ap@0.5S', 'ap@0.5M', 'ap@0.5L', 'ap@0.5', 'ap@0.75', 'ap@0.95']]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.precision', 3)
        pd.set_option('display.width', 120)
        print(df)

    mean = {}

    if metric_fun == voc_eval:
        for kk in ['ap', 'ar', 'f2']:
            k = kk + '@0.5'
            if k in next(iter(evaluation.values())):
                mean[k] = np.mean([v[k] for v in evaluation.values()])
                if print_out:
                    print('* {} = {:.4f}'.format(k, mean[k]))
        for k in ['T']:
            if k in next(iter(evaluation.values())):
                mean[k] = np.mean([v[k] for v in evaluation.values()])
                if print_out:
                    print('* {} = {:.4f}'.format(k, mean[k]))

        assert mean['T'] > 0 # there is no positive sample in dataset!

        evaluation['mean'] = mean
        evaluation['score'] = mean['ap@0.5']

    if output:
        with output.open('wb') as output_file:
            print('save output to {}'.format(output))
            pickle.dump(evaluation, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    return evaluation

