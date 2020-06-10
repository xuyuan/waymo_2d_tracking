
import warnings

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from .modules.prior_box import PriorBox
from .modules.pooling import AdaptiveConcatPool2d
from .modules.detection import Detect
from .modules.activation import XYSigmoid
from .modules.l2norm import L2Norm, l2norm
from .modules.pyramid_feature import lateral_block, ThreeWayBlock, lateral_decoder
from .modules.receptive_field_block import ReceptiveFieldBlockA, ReceptiveFieldBlockB
from .modules.danet import danet_head
from .modules.decoder import decoder
from .modules.det_head import MultiLevelHead, SharedHead, EfficientHead
from .modules.bifpn import bifpn, BiFPN
from ..transforms.test_transforms import TestTransform, torchvision_mean, torchvision_std
# from transforms.train_transforms import TransformOnName
from .basenet import ConvBnRelu, get_num_of_channels, create_basenet, BASENET_CHOICES, load_from_file_or_model_zoo
from .basenet.basic import conv, sequential, conv_relu, relu_conv, conv_bn, maxpool, separable_conv2d
from .basenet.batch_norm import FrozenBatchNorm2d
from .loss.ssd import SSDLoss
from ..utils.box_utils import nms, point_form


def sliding_window(pil_image, size, step):
    for upper in range(0, pil_image.height, step):
        for left in range(0, pil_image.width, step):
            right = left + size
            lower = upper + size
            image = pil_image.crop((left, upper, right, lower))
            yield left, upper, image


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
# feature_maps * steps == min_dim
# num_of_boxes = sum(feature_maps ** 2 * (2 + len(aspect_ratios) * 2))
#              =   38 ** 2 * (2 + 2 * 1)  # 5776
#                 +19 ** 2 * (2 + 2 * 2)  # 2166
#                 +10 ** 2 * (2 + 2 * 2)  # 600
#                 + 5 ** 2 * (2 + 2 * 2)  # 150
#                 + 3 ** 2 * (2 + 2 * 1)  # 36
#                 + 1 ** 2 * (2 + 2 * 1)  # 4
#             = 8732
ssd300 = {
    #'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'features': [l2norm(20)],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
    'extras': [sequential(conv(128, 1, 1, 0), conv(256, 3, 2, 1)),   #(out_channles, kernel_size, stride, padding)
               sequential(conv(128, 1, 1, 0), conv(256, 3, 1, 0)),   #
               sequential(conv(128, 1, 1, 0), conv(256, 3, 1, 0))],  #
}


Retina = {
    'min_dim': 800,
    'min_sizes': [32, 64, 128, 256, 512],
    'scales': [[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]] * 5,
    'aspect_ratios': [[2]] * 5,
    'extras': [sequential(conv(256, 3, 2, 1)),  # 2**6
               sequential(relu_conv(256, 3, 2, 1))],  # 2**7
    'variance': [0.1, 0.2],
    'pyramid_feature': [conv(256, 1, 1, 0)] + [lateral_block(256)] * 4,# groups makes it depthwise
    'class_head': sequential(*([conv_relu(256, 3, 1, 1)] * 4)),
    'bbox_head': sequential(*([conv_relu(256, 3, 1, 1)] * 4)),
    'head_type': SharedHead
 }


def efficient_det(scaling, min_sizes=(32, 64, 128, 256, 512)):
    # Rinput = 512 + scaling * 128
    Rinput = [512, 640, 768, 896, 1024, 1280, 1280, 1536][scaling]
    Dbox = Dclass = 3 + scaling // 3
    #Wbifpn = int(64 * (1.35 ** scaling))
    Wbifpn = [64, 88, 112, 160, 224, 288, 384, 384][scaling]
    Wpred = Wbifpn
    # Dbifpn = 3 + scaling
    Dbifpn = [3, 4, 5, 6, 7, 7, 8, 8][scaling]
    output = lambda out_channels: separable_conv2d(out_channels, 3, 1, 1)
    num_scales = 3

    if scaling == 7:
        min_sizes = [s // 4 * 5 for s in min_sizes]

    # Use unweighted sum for training stability.
    bifpn_weighted_sum = scaling < 6

    coco_pretrained = ['efficientdet-d0-8e11e9ac.pth',
                       'efficientdet-d1-c1d62e9f.pth',
                       'efficientdet-d2-b95f42a4.pth',
                       'efficientdet-d3-54a4ba91.pth',
                       'efficientdet-d4-ee4a07bc.pth',
                       'efficientdet-d5-649e4a4b.pth',
                       'efficientdet-d6-e4590ddc.pth',
                       'efficientdet-d7-4a6e0952.pth']
    
    return {'min_dim': Rinput,
            'min_sizes': min_sizes,
            'scales': [[2 ** (s / num_scales) for s in range(num_scales)]] * len(min_sizes),
            'aspect_ratios': [[(1.4, 0.7)]] * len(min_sizes),
            'extras': [sequential(conv_bn(Wbifpn, 1), maxpool(3, 2, padding='same')),  # 2**6
                       sequential(maxpool(3, 2, padding='same'))],  # 2**7
            'variance': [1, 1],
            'pyramid_feature': bifpn(out_channels=Wbifpn, stack=Dbifpn, weighted_sum=bifpn_weighted_sum),
            'class_head': sequential(*([separable_conv2d(Wpred, 3, 1, 1)] * Dclass)),  # groups --> depthwise
            'class_activation': 'sigmoid',
            'bbox_head': sequential(*([separable_conv2d(Wpred, 3, 1, 1)] * Dbox)),
            'head_type': EfficientHead,
            'bbox_output': output,
            'class_output': output,
            'dropout': 0,
            'pretrained': {'coco': 'model_zoo:' + coco_pretrained[scaling]}
            }


def efficientDet(scaling):
    return {**efficient_det(scaling),
            'class_activation': 'softmax',
            }

# register configurations
SSD_CONFIG = {
    'v2': ssd300,
    'ssd300': ssd300,
    'retina': Retina,
}

for i in range(8):
    SSD_CONFIG[f'efficientdet-d{i}'] = efficient_det(i)
    SSD_CONFIG[f'efficientdet-D{i}'] = efficientDet(i)


class FPN(nn.ModuleList):
    def __init__(self, pyramid_feature, backbone):
        pfn = []
        for i, pfb in enumerate(pyramid_feature):
            if i == 0:
                pfn.append(pfb(backbone[-i - 1]))
            else:
                pfn.append(pfb(backbone[-i - 1], pfn[-1]))
        super().__init__(pfn)

    def forward(self, activations):
        p = self[0](activations[-1])
        activations[-1] = p
        for i in range(1, len(self)):
            p = self[i](activations[-i - 1], p)
            activations[-i - 1] = p
        return activations


class SingleShotDetector(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    IMPLEMENTATION_LATEST = 1
    # BREAK CHANGES OF DIFFERENT IMPLEMENTATION VERSION
    # 1. forward return tensor in N,C,L shape instead of N,L,C

    def __init__(self, classnames, basenet='vgg16', version='v2',
                 pretrained=False, frozen_bn=False,
                 implementation=IMPLEMENTATION_LATEST, state_dict=None):
        super(SingleShotDetector, self).__init__()
        self.classnames = classnames
        self.basenet = basenet
        self.version = version
        self.implementation = implementation

        cfg = SSD_CONFIG[version]
        self.cfg = cfg
        self.min_dim = cfg['min_dim']
        self.global_features = cfg.get('global_features', 0)
        pyramid_feature = cfg.get('pyramid_feature', None)
        head_type = cfg.get('head_type', MultiLevelHead)
        self.mask_output = cfg.get('mask_output', False)
        self.elevation_output = cfg.get('elevation_output', False)
        self.feature_layer_index = cfg.get('feature_layer_index', None)
        self.dropout = cfg.get('dropout', 0)
        xy_sigmoid = cfg.get('xy_sigmoid', False)
        drop_basenet_last_n_layer = cfg.get('drop_basenet_last_n_layer', 0)

        class_activation = cfg.get('class_activation', 'softmax')
        self.num_classes = len(classnames)
        if class_activation == 'softmax':
            self.num_classes += 1

        # build ssd
        prior_box_order = 'hwk' if self.implementation == 0 else 'khw'
        self.priorbox = PriorBox(cfg, prior_box_order)
        nbox_cfg = self.priorbox.nbox_per_layer()
        basenet_pretrained = 'imagenet' if pretrained == 'imagenet' else False
        base, bn, num_of_pretrained_layer = create_basenet(basenet, basenet_pretrained,
                                                           drop_last=drop_basenet_last_n_layer,
                                                           frozen_batchnorm=frozen_bn)
        extra_layers = add_extras(base[-1], cfg.get('extras', []))
        base += extra_layers

        # feature network
        self.backbone = nn.ModuleList(base)
        self.num_of_pretrained_layers = num_of_pretrained_layer

        if self.global_features:
            self.layers_global_features = nn.Sequential(ConvBnRelu(base[-1].out_channels, self.global_features, 1),
                                                        ConvBnRelu(self.global_features, self.global_features // 2, 1),
                                                        AdaptiveConcatPool2d(1)
                                                        )

        # top-down pyramid features
        self.pfn = None
        if pyramid_feature:
            if callable(pyramid_feature):
                self.pfn = pyramid_feature(self.backbone[-self.priorbox.n_layer:])
            else:
                self.pfn = FPN(pyramid_feature, self.backbone)

        head = multibox(base[-self.priorbox.n_layer:], cfg, nbox_cfg, self.num_classes,
                        global_features=self.global_features,
                        pyramid_feature=self.pfn,
                        xy_sigmoid=xy_sigmoid,
                        num_points_per_box=self.priorbox.num_points_per_box())

        self.feature_layers = nn.ModuleList(head[0])
        self.loc = head_type(head[1])
        self.conf = head_type(head[2])

        if self.mask_output:
            #in_channels += self.global_features
            mask_input = self.pfn[-1] if pyramid_feature else self.backbone[-len(self.feature_layers)]
            decoders = self.mask_output(mask_input)
            self.mask_head = nn.Sequential(decoders,
                                           nn.Conv2d(decoders.out_channels, self.num_classes, 1))

        if self.elevation_output:
            mask_input = self.pfn[-1] if pyramid_feature else self.backbone[-len(self.feature_layers)]
            decoders = self.elevation_output(mask_input)
            self.elevation_head = nn.Sequential(decoders,
                                                nn.Conv2d(decoders.out_channels, self.num_classes - 1, 1),
                                                #nn.Hardtanh(min_val=0, max_val=1, inplace=False)
                                                )

        self.detector = Detect(cfg['variance'], class_activation=class_activation)

        self.init_weights()

        if pretrained in ('coco', 'oid'):
            model_url = cfg.get('pretrained', {}).get(pretrained, None)
            if not model_url:
                raise NotImplementedError(f'No {pretrained} pretrained weights for {version}')
            pretrained_state_dict = load_from_file_or_model_zoo(model_url)
            print('Loading weights from: {}'.format(model_url))
            self.load_part_state_dict(pretrained_state_dict['state_dict'])

            if frozen_bn:
                FrozenBatchNorm2d.convert_frozen_batchnorm(self)

        if state_dict:
            self.load_state_dict(state_dict)
        self.trace_mode = False
        self.num_of_frozen_pretrained_layers = 0

        self.test_transform = TestTransform(self.min_dim, torchvision_mean, torchvision_std)


    @property
    def input_mean(self):
        return self.test_transform.preprocessor.mean

    @property
    def input_std(self):
        return self.test_transform.preprocessor.std

    def set_pretrained_frozen(self, num_of_frozen_layers):
        for i in range(self.num_of_pretrained_layers):
            for param in self.backbone[i].parameters():
                param.requires_grad = (i >= num_of_frozen_layers)
        self.num_of_frozen_pretrained_layers = num_of_frozen_layers
        print(f'freeze {self.num_of_frozen_pretrained_layers} layers')

    def train(self, mode=True):
        ret = super().train(mode)
        # set dropout and BN state for freezing pretrained
        if mode:
            for i in range(self.num_of_frozen_pretrained_layers):
                self.backbone[i].eval()
        return ret

    def predict_slide_window(self, pil_image, tta=None, image_size=None, conf_thresh=0.01, nms_thresh=0.15, max_bbox=0, mask_thresh=-1, mask2rbox=False, **kwargs):
        """predict using sliding windows"""
        if image_size is None:
            image_size = self.min_dim

        if torch.is_tensor(pil_image):
            pil_image.squeeze_(dim=0)
            assert pil_image.dim() == 3
            pil_image = Image.fromarray(pil_image.numpy())

        detections = []
        if pil_image.width <= image_size and pil_image.height <= image_size:
            detections= self._predict(pil_image=pil_image, tta=tta, image_size=image_size, conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_bbox=max_bbox,
                                      mask_thresh=mask_thresh, mask2rbox=mask2rbox, **kwargs)
        else:
            step_size = image_size * 3 // 4  # 1/4 overlap

            for left, upper, img in sliding_window(pil_image, image_size, step_size):
                det = self._predict(pil_image=img, tta=tta, image_size=image_size, conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_bbox=max_bbox,
                                    mask_thresh=mask_thresh, mask2rbox=mask2rbox, **kwargs)

                scale_height = img.height / pil_image.height
                scale_width = img.width / pil_image.width
                left /= pil_image.width
                upper /= pil_image.height
                scale = det[0].new([scale_width, scale_height, scale_width, scale_height])

                for c, d in enumerate(det):
                    # TODO: optimize
                    d[:, 1:5] *= scale
                    d[:, 1] += left
                    d[:, 2] += upper
                    if len(detections) > c:
                        detections[c].append(d)
                    else:
                        detections.append([d])
            detections = [torch.cat(d, dim=0) for d in detections]
            # NMS
            detections = [d[nms(point_form(d[:, 1:5]), d[:, 0], nms_thresh)[0]] for d in detections]

        detections = [d.cpu().numpy() for d in detections]
        return detections

    def predict(self, *args, **kwargs):
        detections = self._predict(*args, **kwargs)
        detections = [d.cpu().numpy() for d in detections]
        return detections

    def _predict(self, pil_image, tta=None, image_size=None, conf_thresh=0.01, nms_thresh=0.15, max_bbox=0, mask_thresh=-1, mask2rbox=False, **kwargs):
        """predict in single pass"""
        with torch.no_grad():
            x, height_width = self.test_transform.pre_process(pil_image, min_dim=image_size, data_aug=tta)
            # assuming model is on a single device
            # https://github.com/pytorch/pytorch/issues/584
            net_param = next(self.parameters())
            x = x.to(net_param)

            y = self.forward(x)
            loc, conf = y[:2]
            priors = self.priorbox.priors
            bboxes = self.detector.detect(loc, conf, priors, conf_thresh, nms_thresh, **kwargs)

            if False and self.mask_output and self.elevation_output:
                mask = y[2]
                elevation = y[3]
                detections = self.test_transform.post_process_mask_with_elevation(height_width, mask, elevation, bboxes, tta,
                                                                                  conf_thresh, nms_thresh)
            elif self.mask_output:
                mask = y[2]
                detections = self.test_transform.post_process_mask(height_width, mask, bboxes, tta,
                                                                   conf_thresh, nms_thresh, mask_thresh, mask2rbox)
            else:
                detections = self.test_transform.post_process(height_width, bboxes, tta, conf_thresh, nms_thresh, max_bbox=max_bbox)
            return detections

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch, 3, H, W].

        Return:
            list of concat outputs from:
                1: confidence layers, Shape: [batch, num_classes, num_priors]
                2: localization layers, Shape: [batch, 4, num_priors]
        """
        h = x.size(2)
        w = x.size(3)
        image_size = (h, w)
        if (not self.trace_mode) and (h < self.min_dim or w < self.min_dim):
            warnings.warn(
                "input image is too small {}x{}, minimal required size is {}x{}.".format(h, w, self.min_dim, self.min_dim))

        y = self.pretrained_forward(x)
        y = self.unpretrained_forward(x, y, image_size)
        return y

    def pretrained_forward(self, x):
        """pretrained network"""

        y = list()
        for i in range(self.num_of_pretrained_layers):
            layer = self.backbone[i]
            x = layer(x)
            y.append(x)
        return y

    def unpretrained_forward(self, x, activations, image_size):
        """untrained network"""
        if activations:
            x = activations[-1]

        for i in range(self.num_of_pretrained_layers, len(self.backbone)):
            layer = self.backbone[i]
            x = layer(x)
            activations.append(x)

        if self.pfn:
            activations = self.pfn(activations)

        if self.feature_layer_index is None:
            # take last layers
            features = activations[-self.priorbox.n_layer:]
        else:
            # remap feature layers
            features = [activations[i] for i in self.feature_layer_index]

        if self.global_features:
            gf = self.layers_global_features(x)
            for k, x in enumerate(features):
                gfx = gf.expand(-1, -1, x.size(2), x.size(3))
                features[k] = torch.cat((x, gfx), dim=1)

        if self.mask_output:
            mask = self.mask_head(features[0])
        else:
            mask = None

        if self.elevation_output:
            elevation = self.elevation_head(features[0])
        else:
            elevation = None

        # apply multibox head to source layers
        for k, layer in enumerate(self.feature_layers):
            features[k] = layer(features[k])

        if self.dropout > 0 and self.training:
            features = [torch.nn.functional.dropout2d(i, p=self.dropout, training=self.training) for i in features]

        loc = self.loc(features)
        conf = self.conf(features)

        #self.feature_maps = conf  # for receptivefield.pytorch

        if not self.training and not self.trace_mode:
            self.priorbox((image_size, loc))

        num_points_per_box = self.priorbox.num_points_per_box()
        if self.implementation == 0:
            # LxN(AC)HW -> LxNHW(AC)
            loc = [l.permute(0, 2, 3, 1).contiguous() for l in loc]
            conf = [c.permute(0, 2, 3, 1).contiguous() for c in conf]

            # --> LxN(HWA)C --> N(LHWA)C
            loc = torch.cat([o.view(o.size(0), -1, num_points_per_box) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

            # --> NC(LHWA)
            loc = loc.transpose(1, 2)
            conf = conf.transpose(1, 2)
        else:
            # LxN(CA)HW --> LxNC(AHW) --> NC(LAHW)
            loc = torch.cat([o.view(o.size(0), num_points_per_box, -1) for o in loc], dim=-1)
            conf = torch.cat([o.view(o.size(0), self.num_classes, -1) for o in conf], dim=-1)

        output = [loc, conf]
        if not self.trace_mode:
            output += [mask, elevation]
        else:
            output += [o for o in [mask, elevation] if o is not None]

        return output

    def load_weights(self, model_file):
        print('Loading weights from', model_file)
        data = load_from_file_or_model_zoo(model_file)
        self.load_part_state_dict(data['state_dict'])

    def load_part_state_dict(self, pretrained_dict):
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        missing_keys = []
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                missing_keys.append(k)
            else:
                if pretrained_dict[k].size() == v.size():
                    # 2. overwrite entries in the existing state dict
                    model_dict[k] = pretrained_dict[k]
                else:
                    warnings.warn(f"'{k}' has shape {pretrained_dict[k].size()} in the checkpoint but {model_dict[k].size()} in the model! Skipped.")
        if missing_keys:
            warnings.warn(f'Missing key(s) in state_dict: {missing_keys}. ')

        # 3. load the new state dict
        self.load_state_dict(model_dict)
        # NOTE: assume backbone is fully pretrained
        self.num_of_pretrained_layers = len(self.backbone)

    def init_weights(self):
        """
        @note the pretrained layers are not changed here
        """
        for i in range(self.num_of_pretrained_layers, len(self.backbone)):
           self.backbone[i].apply(weights_init)

        if self.pfn:
           self.pfn.apply(weights_init)

        for feature_layers in self.feature_layers:
           feature_layers.apply(weights_init)

        # weights init as "official"
        class_activation = self.cfg.get('class_activation', 'softmax')
        self.conf.bias_init_with_prior_prob(activation=class_activation)

        #if self.global_features:
        #    self.layers_global_features.apply(weights_init)

    def full_state_dict(self):
        return dict(state_dict=self.state_dict(),
                    classes=self.classnames,
                    basenet=self.basenet,
                    version=self.version,
                    input_mean=self.input_mean,
                    input_std=self.input_std,
                    implementation=self.implementation
                    )

    def save(self, filename):
        data = self.full_state_dict()
        torch.save(data, filename)

    @staticmethod
    def load_from_file_or_model_zoo(filename):
        data = load_from_file_or_model_zoo(filename)

        if 'implementation' not in data:
            # missing in the earliest models
            data['implementation'] = 0

        return data

    @staticmethod
    def load(filename):
        data = SingleShotDetector.load_from_file_or_model_zoo(filename)
        return SingleShotDetector(pretrained=False, **data)

    # def preprocessed_dataset(self, dataset):
    #     """warp dataset with preprocessor"""
    #     transforms = TransformOnName(self.test_transform.preprocessor)
    #     return TransformedDataset(dataset, transforms)

    def criterion(self, args):
        class_activation = self.cfg.get('class_activation', 'softmax')
        return SSDLoss(self.priorbox.priors, self.cfg['variance'], self.num_classes, class_activation, args)


def weights_init(m, conv_bias=0):
    if isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal(m.weight.data)
        nn.init.kaiming_uniform_(m.weight.data)
        # nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(conv_bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def add_extras(x, cfg):
    # Extra layers added to basenet for feature scaling
    layers = []
    for layer_cfg in cfg:
        x = layer_cfg(x)
        layers.append(x)
    return layers


def head_layer(k, nbox_cfg, in_channels, num_classes, xy_sigmoid, num_points_per_box, cfg):
    feature_layers_cfg = cfg.get('features', None)
    feature, f_channels = create_features_block(k, feature_layers_cfg, in_channels)

    class_head_cfg = cfg.get('class_head', None)
    bbox_head_cfg = cfg.get('bbox_head', None)
    output_default = lambda out_channels: conv(out_channels, kernel_size=3, padding=1)
    bbox_output = cfg.get('bbox_output', output_default)
    class_output = cfg.get('class_output', output_default)

    nbox = nbox_cfg[k]
    bbox_header, bbox_channels = create_features_block(k, bbox_head_cfg, f_channels)
    loc = bbox_output(nbox * num_points_per_box)(bbox_channels)
    if xy_sigmoid:
        loc = nn.Sequential(loc, XYSigmoid())
    if bbox_header:
        loc = nn.Sequential(bbox_header, loc)

    class_head, class_channels = create_features_block(k, class_head_cfg, f_channels)
    conf = class_output(nbox * num_classes)(class_channels)
    if class_head:
        conf = nn.Sequential(class_head, conf)
    return feature, loc, conf


def multibox(base, cfg, nbox_cfg, num_classes, global_features, pyramid_feature, xy_sigmoid, num_points_per_box):
    loc_layers = []
    conf_layers = []
    feature_layers = []

    feature_layer_index = cfg.get('feature_layer_index', None)

    if pyramid_feature:
        if isinstance(pyramid_feature, BiFPN):
            features = [pyramid_feature.out_channels] * pyramid_feature.levels
        else:
            # reversed order
            features = [pyramid_feature[i - 1] for i in range(len(pyramid_feature))]
    else:
        features = base

    if feature_layer_index is not None:
        features = [base[i] for i in feature_layer_index]

    for k, v in enumerate(features):
        in_channels = get_num_of_channels(v)
        in_channels += global_features
        feature, loc, conf = head_layer(k, nbox_cfg, in_channels, num_classes, xy_sigmoid, num_points_per_box, cfg=cfg)
        if feature:
            feature_layers.append(feature)
        loc_layers.append(loc)
        conf_layers.append(conf)
    return feature_layers, loc_layers, conf_layers


def create_features_block(k, cfg, in_channels):
    """feature block between backbone and heads"""
    if isinstance(cfg, list) or isinstance(cfg, tuple):
        if len(cfg) > k:
            cfg = cfg[k]
        else:
            cfg = None

    if cfg is None:
        return None, in_channels

    if callable(cfg):
        layers = cfg(in_channels)
        return layers, layers.out_channels

    layers = []
    for c in cfg:
        layer = c(in_channels)
        in_channels = layer.out_channels
        layers.append(layer)
    return nn.Sequential(*layers), in_channels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="show network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='', help='load saved model')
    parser.add_argument('--version', default='v2', choices=SSD_CONFIG.keys(), help='variants version')
    parser.add_argument('--basenet', default='vgg16', choices=BASENET_CHOICES, help='base net for feature extracting')
    parser.add_argument('--classes', default=1, type=int, help='number of classes')
    parser.add_argument("-v", "--verbose", action='count', default=0, help="level of debug messages")

    args = parser.parse_args()

    if args.model:
        net = SingleShotDetector.load(args.model)
    else:
        classnames = ['bg'] + [str(i) for i in range(args.classes)]
        net = SingleShotDetector(classnames, basenet=args.basenet, version=args.version)
    # print(repr(net))
    if args.verbose > 0:
        print(net)
    parameters = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in parameters)
    print('N of parameters {} ({})'.format(n_params, len(parameters)))

    x = torch.empty((1, 3, net.min_dim, net.min_dim))
    net.eval()
    y = net(x)
    print(x.size(), '->', net.priorbox.priors_size)
    print(net.priorbox.priors.size())
    print([i.size() for i in y if i is not None])

