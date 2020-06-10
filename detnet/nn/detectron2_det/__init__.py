from pathlib import Path
import torch
import numpy as np
from torch.nn import Module
from detectron2.model_zoo.model_zoo import get_config_file, get_cfg, get_checkpoint_url, build_model, DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import MetadataCatalog
from PIL import Image


def get_local_config_file(config_path):
    local_configs = Path(__file__).parent / "configs"
    local_file = local_configs / config_path
    if local_file.exists():
        return str(local_file)
    return None


def model_zoo_get(config_path, classnames: int, trained: bool = False, freeze_at: int = 2, frozen_bn: bool = True):
    """replicated `model_zoo.get` with additional config modification"""
    cfg_file = get_local_config_file(config_path)

    model_zoo_cfg = False
    if not cfg_file:
        cfg_file = get_config_file(config_path)
        model_zoo_cfg = True

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)

    if cfg.MODEL.META_ARCHITECTURE == "PanopticFPN":
        print("change PanopticFPN to GeneralizedRCNN")
        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

    if trained and model_zoo_cfg:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    cfg.MODEL.RESNETS.NORM = "FrozenBN" if frozen_bn else "BN"

    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.5]

    if classnames:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classnames)
    else:
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        classnames = meta.thing_classes

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.MASK_ON = False

    model = build_model(cfg)
    if trained:
        print(f'load pretrained weights {cfg.MODEL.WEIGHTS}')
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return cfg, model, classnames


class Detectron2Det(Module):
    def __init__(self, arch, classnames, freeze_pretrained=2, frozen_bn=True, pretrained=True):
        super().__init__()
        trained = pretrained == 'coco'

        self.cfg, self.model, self.classnames = model_zoo_get(arch, classnames, trained=trained, freeze_at=freeze_pretrained, frozen_bn=frozen_bn)

    def forward(self, x):
        # compute everything in loss
        if self.cfg.INPUT.FORMAT == 'BGR':
            x = x[:, [2, 1, 0]]  # BGR input
        return x

    def predict(self, x):
        """
        When in inference mode, the builtin models output a list[dict], one dict for each image. Based on the tasks the
         model is doing, each dict may contain the following fields:

        “instances”: Instances object with the following fields:
            “pred_boxes”: Boxes object storing N boxes, one for each detected instance.
            “scores”: Tensor, a vector of N scores.
            “pred_classes”: Tensor, a vector of N labels in range [0, num_categories).
            “pred_masks”: a Tensor of shape (N, H, W), masks for each detected instance.
            “pred_keypoints”: a Tensor of shape (N, num_keypoint, 3). Each row in the last dimension is (x, y, score).
             Scores are larger than 0.

        “sem_seg”: Tensor of (num_categories, H, W), the semantic segmentation prediction.

        “proposals”: Instances object with the following fields:
            “proposal_boxes”: Boxes object storing N boxes.
            “objectness_logits”: a torch vector of N scores.

        “panoptic_seg”: A tuple of (Tensor, list[dict]). The tensor has shape (H, W), where each element represent the
         segment id of the pixel. Each dict describes one segment id and has the following fields:
            “id”: the segment id
            “isthing”: whether the segment is a thing or stuff
            “category_id”: the category id of this segment. It represents the thing class id when isthing==True, and
             the stuff class id otherwise.
        """
        with torch.no_grad():
            single_image_input = False
            if isinstance(x, Image.Image):
                x = np.float32(x)
                x = torch.as_tensor(x.transpose(2, 0, 1))
                x = x.unsqueeze(0)
                single_image_input = True

            net_param = next(self.parameters())
            x = x.to(net_param)
            x = self(x)  # forward

            height, width = x.shape[-2:]
            inputs = [dict(image=i, height=height, width=width) for i in x]
            detections = self.model(inputs)

            output = []
            for det in detections:
                bbox_cls = []
                instances = det['instances']
                if instances.has('scores'):
                    scores = instances.scores.unsqueeze(1)
                    box = instances.pred_boxes
                    box.scale(1 / x.shape[-1], 1 / x.shape[-2])
                    center = box.get_centers()
                    wh = (box.tensor[:, 2:4] - box.tensor[:, 0:2])
                    bbox = torch.cat((scores, center, wh), dim=1)
                    labels = instances.pred_classes
                    for cls in range(len(self.classnames)):
                        bbox_cls.append(bbox[labels == cls].cpu().numpy())
                else:
                    for cls in range(len(self.classnames)):
                        bbox_cls.append(np.empty((0, 5)))
                output.append(bbox_cls)

            if single_image_input:
                output = output[0]
            return output

    def criterion(self, args):
        return self.loss

    def loss(self, images, target):
        """
        All builtin models take a list[dict] as the inputs. Each dict corresponds to information about one image.

        The dict may contain the following keys:

            “image”: Tensor in (C, H, W) format. The meaning of channels are defined by cfg.INPUT.FORMAT.
            Image normalization, if any, will be performed inside the model.

            “instances”: an Instances object, with the following fields:
                “gt_boxes”: a Boxes object storing N boxes, one for each instance.
                “gt_classes”: Tensor of long type, a vector of N labels, in range [0, num_categories).
                “gt_masks”: a PolygonMasks or BitMasks object storing N masks, one for each instance.
                “gt_keypoints”: a Keypoints object storing N keypoint sets, one for each instance.

            “proposals”: an Instances object used only in Fast R-CNN style models, with the following fields:
                “proposal_boxes”: a Boxes object storing P proposal boxes.
                “objectness_logits”: Tensor, a vector of P scores, one for each proposal.

            “height”, “width”: the desired output height and width, which is not necessarily the same as the height or
             width of the image input field. For example, the image input field might be a resized image, but you may
              want the outputs to be in original resolution.

            If provided, the model will produce output in this resolution, rather than in the resolution of the image
            as input into the model. This is more efficient and accurate.

            “sem_seg”: Tensor[int] in (H, W) format. The semantic segmentation ground truth. Values represent category
            labels starting from 0.
        """
        batch_size = len(images)
        inputs = []
        image_size = images.shape[-2:]
        for i in range(batch_size):
            gt_classes = target['labels'][i]
            gt_boxes = Boxes(target['boxes'][i])
            instances = Instances(image_size=image_size, gt_boxes=gt_boxes, gt_classes=gt_classes - 1)

            inputs.append(dict(image=images[i],
                               instances=instances))

        with EventStorage() as storage:
            losses = self.model(inputs)
        return losses

    def enable_tta(self, min_sizes=None):
        if not isinstance(self.model, GeneralizedRCNNWithTTA):
            if min_sizes:
                self.cfg.TEST.AUG.MIN_SIZES = min_sizes
            print(f'TEST.AUG.MIN_SIZES={self.cfg.TEST.AUG.MIN_SIZES}')
            #cfg.TEST.AUG.MAX_SIZE
            self.cfg.TEST.AUG.FLIP = False
            self.model = GeneralizedRCNNWithTTA(self.cfg, self.model)
