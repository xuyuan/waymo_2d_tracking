from .trainer import Trainer
from .trainer import ArgumentParser as BaseArgumentParser
from .trainer.data import EchoingDataset, TransformedDataset
from .trainer.transforms.vision import *
from .transforms.train_transforms import create_match_prior_box, TrainAugmentation, creat_point_heat_map
from .nn import create
from .nn.basenet import BASENET_CHOICES
from .nn.torchvision_det import TorchVisionTrans
# from trainer.utils import warn_with_traceback
# warnings.showwarning = warn_with_traceback


class ArgumentParser(BaseArgumentParser):
    def __init__(self, description=__doc__):
        super().__init__(description)
        self.parser.add_argument('--data-echo', action='store_true', help="echoing training data")

        group = self.parser.add_argument_group('options of model')
        group.add_argument('--arch', default='ssd300', help='model architecture', type=str)
        group.add_argument('--basenet', default='vgg16', choices=BASENET_CHOICES,
                           help='base net for feature extracting')
        group.add_argument('--pretrained', default='coco', choices=('imagenet', 'coco', 'oid'),
                           help='dataset name for pretrained basennet')
        group.add_argument('--freeze-pretrained', default=0, type=int, help='freeze pretrained layers')
        group.add_argument('--frozen-bn', action='store_true', help='freeze BN layers in basenet')

        group = self.parser.add_argument_group('options of loss criterion')
        group.add_argument('--fg-iou-threshold', default=0.5, type=float, help='Min IoU for foreground')
        group.add_argument('--bg-iou-threshold', default=0.5, type=float, help='Max IoU for background')
        group.add_argument('--warn-unmatched-bbox', action='store_true', help='warn when bbox unmatched to any anchor')
        group.add_argument("--loss-focusing", type=float, default=0, help='focusing parameter for Focal Loss')
        group.add_argument("--loss-positive-weight", type=float, default=0,
                           help='scalar multiplying alpha to the loss from positive examples and (1-w) to the loss from negative examples')
        group.add_argument("--loss-iou-weighting", action='store_true', help='weighting loss by IOU')
        group.add_argument("--label-smoothing", type=float, default=0, help='factor of label smoothing')
        group.add_argument("--loss-box-weight", type=float, default=1, help='weight for loss of box regression')
        group.add_argument("--loss-guided", action='store_true', help='scale classification loss by regression loss')
        group.add_argument("--loss-negpos-ratio", type=float, default=3,
                           help='ratio of negitive:positive for Online Hard Example Mining, when value > 0')
        group.add_argument("--loss-box", choices=['SmoothL1', 'AdjustSmoothL1', 'IoU', 'GIoU', 'FocalL1'],
                           default='SmoothL1', help='criterion of box regression')
        group.add_argument("--loss-mask", choices=['CrossEntropy', 'Focal', 'Lovasz'], default='CrossEntropy',
                           help='criterion of mask')

        group.add_argument("--download-pretrained-weights", action='store_true', help='download pretrained weights and exit')


def train(datasets, args):
    if datasets:
        classnames = datasets['test'].classnames
    else:
        classnames = "This is fallback".split()
    trainer = Trainer(args)
    model = create(args.arch, classnames=classnames, basenet=args.basenet, pretrained=args.pretrained,
                   freeze_pretrained=args.freeze_pretrained, frozen_bn=args.frozen_bn)

    if args.download_pretrained_weights:
        print('pretrained weights downloaded')
        exit(0)

    facebook_net = args.arch.split(':')[0] in ('torchvision', 'detectron2', 'mmdet')

    if args.arch == 'pointdet':
        convert_train_target = creat_point_heat_map(model, args.image_size)
    elif facebook_net:
        convert_train_target = TorchVisionTrans()
    else:
        convert_train_target = create_match_prior_box(model, args.image_size, args.bg_iou_threshold, args.fg_iou_threshold,
                                                      return_iou=args.loss_iou_weighting,
                                                      warn_unmatched_bbox=args.warn_unmatched_bbox)

    to_tensor = ToTensor()
    if args.arch.split(':')[0] == 'detectron2' or args.arch.split(':')[0] == 'mmdet':
        to_tensor = ToTensor(scaling=False, normalize=None)

    comm_trans = Compose([ScaleBBox(), to_tensor, convert_train_target])
    datasets['train'] = TransformedDataset(datasets['train'], comm_trans)

    if args.arch in ('pointdet',) or facebook_net:
        datasets['test'] = TransformedDataset(datasets['test'], to_tensor)
    else:
        if args.test_batch_size > 0:
            datasets['test'] = TransformedDataset(datasets['test'], ToByteTensor())

    if args.data_echo:
        datasets['train'] = EchoingDataset(datasets['train'], 64)

    the_criterion = model.criterion(args)

    trainer.run(model, datasets, the_criterion)


if __name__ == '__main__':
    from .data import create_dataset, add_dataset_argument

    parser = ArgumentParser()
    group = add_dataset_argument(parser)
    args = parser.parse_args()

    data_augs = dict(train=TrainAugmentation(args.image_size),
                     test=None)

    datasets = {mode: create_dataset(args.data_root, mode=mode, data_fold=args.data_fold,
                                     split_file=args.data_split_file,
                                     transform=trans)
                for mode, trans in data_augs.items()}

    train(datasets, args)
