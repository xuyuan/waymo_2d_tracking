# Baseline for 2D Object Detection and Tracking: Cascade R-CNN X152 and SORT

This repo is the 3rd place solution (method name: CascadeRCNN-SORT v2) of [Waymo open dataset challenge 2D tracking track](https://waymo.com/open/challenges/2d-tracking/).
See our [report](doc/waymo_2d_tracking_dainamite.pdf) for technical details.


## Installation
Create a conda virtual [environment](environment.yml) and activate it.
```sh
conda env update
conda activate waymo_2d_tracking
```

## Dataset Preprocessing
Download [Waymo open dataset](https://waymo.com/open/download/), extract images from TFRecords and generate annotations.json in COCO format with command:
```sh
python waymo_to_coco.py -i ${WAYMO_DATASET}/training -o ${EXPORTED_DATASET_ROOT}/training -j ${NUM_CPU_CORES} -t ${FRAME_INTERVAL}
```

## Detection

Training with multi GPUs:
```sh
./detnet/trainer/launch.sh train.py @detnet/configs/detectron2.cfg --data-root=${EXPORTED_DATASET_ROOT} --batch-size=1 --batch-size-per-gpu --sync-bn --arch=detectron2:Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml
```
Tensorboard training [logs](logs) of our best model are provided in for reference.

Testing with multi GPUs:
```sh
python inference.py -m ${MODEL_FILE} --batch-size=1 -j=${N_GPUS} -o test_output --export $SUBMISSION -i ${PATH_TO_TESTING_IMAGES_FOLDER}
```
TTA can be applied with args, e.g.: `--tta x1.5,hflip --auto-contrast=1`

Ensemble multi testings:
```sh
python -m detnet.ensemble $SUBMISSION1 SUBMISSION2 [SUBMISSION3 ...] -o  $SUBMISSION_ENSEMBLED -m soft_nms --min-score=0.01 --soft-nms-cut=0.9 -j -1
```

Make submission for official evaluation:
```sh
python coco_to_waymo.py --unique-method-name ${METHOD_NAME} --description ${DESCRIPTION} ${SUBMISSION.json} -o submission.bin
```
The [detection results](detection_results.md) of our submissions are provided for reference.

## Tracking

For tracking, we use [SORT](https://github.com/abewley/sort) with optimized score and IOU thresholds. To create a tracking submission:
```sh
python tracking/track.py --input $SUBMISSION_ENSEMBLED --output tracking_submission.json --max-age=2 --min-hits=0 --score-threshold=0.95,0.6,1.0,0.9
python coco_to_waymo.py --unique-method-name ${METHOD_NAME} --description ${DESCRIPTION} tracking_submission.json -o submission.bin
```

