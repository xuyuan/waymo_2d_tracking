"""
run evaluation on test dataset
"""

import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from .data.coco import category_count


def coco_eval(cocoGt, cocoDt, catIds):
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')  # initialize CocoEval object
    if catIds:
        E.params.catIds = catIds
        print([cocoGt.cats[c] for c in E.params.catIds])
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument("--gt", type=str, help='ground truth annotation')
    parser.add_argument('--dt', type=str, default='submission.json', help='detection json')
    parser.add_argument('--dt-images', action='store_true', help='use images of detection only')
    parser.add_argument('--catId', type=lambda s: [int(i) for i in s.split(',')], default=[], help='select catIds')
    parser.add_argument('--each-category', action='store_true', help='evaluate each category')
    args = parser.parse_args()

    cocoGt = COCO(args.gt)

    if args.dt_images:
        dt = json.load(open(args.dt))
        dt_images = set([d['image_id'] for d in dt])
        print('#images:', len(dt_images))
        cocoGt.imgs = {k: v for k, v in cocoGt.imgs.items() if k in dt_images}

    cocoDt = cocoGt.loadRes(args.dt)

    catIds = cocoGt.getCatIds(catIds=args.catId)
    if args.each_category:
        cat_count = category_count(cocoGt)
        print('cat_count', cat_count)
        cat_eval = {c: coco_eval(cocoGt, cocoDt, [c]) for c in catIds}
        cat_eval = {c: v for c, v in cat_eval.items() if (v >= 0).any()}
        columns = ['mAP', 'AP50', 'AP75', 'mAPs', 'mAPm', 'mAPl', 'AR1', 'AR10', 'AR', 'ARs', 'ARm', 'ARl', 'T', 'name']
        values = [v.tolist() + [cat_count[k], cocoGt.cats[k]['name']] for k, v in cat_eval.items()]
        cat_eval = pd.DataFrame(values, index=cat_eval.keys(), columns=columns)
        print("=============================================== summary ===============================================")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.precision', 3)
        pd.set_option('display.width', 120)
        print(cat_eval.sort_values(by=['mAP']))

    coco_eval(cocoGt, cocoDt, catIds)

