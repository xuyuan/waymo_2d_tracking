from .__init__ import BASENET_CHOICES, create_basenet

if __name__ == '__main__':
    import argparse
    from torchsummary import summary
    import torch
    from torch import nn
    parser = argparse.ArgumentParser(description="show network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', nargs=1, choices=BASENET_CHOICES, help='load saved model')
    parser.add_argument('--pretrained', default=False, type=str, choices=('imagenet', 'instagram', 'voc', 'coco', 'oid'), help='pretrained dataset')
    parser.add_argument('--activation', type=str, choices=('mish', 'relu', 'swish', 'hardswish'), help='convert activation functions')
    parser.add_argument('--frozen-batchnorm', action='store_true', help='Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.')
    parser.add_argument('--input', type=lambda s: tuple(int(v) for v in s.split('x')), help='input image size, e.g. 256x128')
    parser.add_argument("-v", "--verbose", action='count', default=0, help="level of debug messages")

    args = parser.parse_args()
    layers, bn, n_pretrained = create_basenet(args.model[0], args.pretrained, activation=args.activation,
                                              frozen_batchnorm=args.frozen_batchnorm)
    model = nn.Sequential(*(layers))

    in_channels = 3
    if args.verbose == 1:
        print(model)

    if args.verbose == 2:
        summary(model, input_size=(in_channels, 512, 512), device='cpu')

    if args.verbose == 3:
        for i, l in enumerate(layers):
            print(f"**{i}**")
            summary(l, input_size=(in_channels, 32, 32), device='cpu')
            in_channels = l.out_channels

    if args.input:
        x = torch.zeros([1, 3, args.input[0], args.input[1]])
        model.eval()
        y = model(x)
        print(x.shape, '-->', y.shape)
