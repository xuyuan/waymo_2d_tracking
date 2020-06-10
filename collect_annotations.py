"""collect annotations in subfolders"""

import argparse
from pathlib import Path
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, required=True, help='path to root folder')
    args = parser.parse_args()

    root = Path(args.root)
    output_json = root / 'annotations.json'
    assert not output_json.exists()

    images = []
    annotations = []
    categories = []

    annotation_files = root.glob("**/annotations.json")
    for anno_file in annotation_files:
        print('read', anno_file)
        parent = anno_file.relative_to(root).parent
        data = json.load(anno_file.open())
        for image in data['images']:
            image['file_name'] = str(parent / image['file_name'])

        images += data['images']
        annotations += data['annotations']
        categories = data['categories']  # assume all categories are the same

    with output_json.open('w') as f:
        print('save', output_json)
        for i, anno in enumerate(annotations):
            anno['id'] = i
            anno['iscrowd'] = 0
            anno['segmentation'] = []
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)
