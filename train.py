from detnet.train import ArgumentParser, train
from detnet.transforms.train_transforms import TrainAugmentation, Compose, Crop
# from trainer.utils import warn_with_traceback
# warnings.showwarning = warn_with_traceback


if __name__ == '__main__':
    from data import create_dataset, add_dataset_argument

    parser = ArgumentParser()
    group = add_dataset_argument(parser)
    group.add_argument('--data-train-all', action='store_true', help='use all data in training')
    args = parser.parse_args()

    image_size = 886, 1280
    datasets = {mode: create_dataset(args.data_root, mode=mode, ingore_empty=not args.data_include_empty,
                                     bgr=args.data_bgr)
                for mode in ('train', 'test')}

    if args.data_train_all:
        datasets['train'] = datasets['train'] + datasets['test']

    data_aug = Compose([Crop(image_size, center=(image_size[1]//2, -image_size[0]//2)),
                        TrainAugmentation(image_size)])
    datasets['train'] = datasets['train'] >> data_aug

    train(datasets, args)
