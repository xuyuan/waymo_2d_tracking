if __name__ == '__main__':
    from pathlib import Path
    from detnet.inference import ArgumentParser, inference, ImageFolder
    from data import create_dataset, add_dataset_argument, Dataset, ToBGR
    parser = ArgumentParser()
    add_dataset_argument(parser.parser)
    args = parser.parse_args()

    if args.input:
        input_root = Path(args.input)
        anno_file = input_root / "annotations.json"
        if input_root.exists() and input_root.is_dir() and anno_file.exists():
            dataset = Dataset(input_root, anno_file)
        else:
            dataset = ImageFolder(input_root)
    else:
        dataset = create_dataset(args.data_root, mode='test')

    if args.data_bgr:
        dataset = dataset >> ToBGR()

    inference(dataset, args)
