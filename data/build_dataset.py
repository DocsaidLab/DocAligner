import re
from pathlib import Path

import docsaidkit as D

DIR = D.get_curdir(__file__)

ROOT = Path('/data/Dataset')

MIDV_ROOT = str(ROOT / 'midv_dataset')

INDOOR_ROOT = str(ROOT / 'indoor_scene_recognition')

CORD_ROOT = str(ROOT / 'cord_v0')


def build_midv_dataset():
    """
    Build a dataset for MIDV-500/2019 dataset.

    This function reads images and corresponding ground-truth files from
    MIDV_ROOT, processes them, and stores the resulting dataset in a JSON file
    named 'midv_dataset.json'.

    The MIDV-500/2019 dataset contains images in the .tif format with filenames
    of the form 'XX00_00.tif', where 'XX' represents two uppercase letters, '00'
    represents two digits, and '_00' represents two more digits.

    The function performs the following steps:
        1. Read all .tif image files from the MIDV_ROOT directory.
        2. Extract information from the filenames to identify the imaging type
            and person ID.
        3. Check if the filename matches the expected pattern, otherwise skip
            the file.
        4. Create a dataset dictionary containing information about each person
            and their corresponding images and ground-truth paths.
        5. Store the dataset in the 'midv_dataset.json' file.
    """
    dataset = {}
    imaging_type_pattern = re.compile(r'^[A-Z]{2}\d{2}_\d{2}$')
    midv_fs = D.get_files(MIDV_ROOT, suffix=['.tif'])
    for f in D.Tqdm(midv_fs):

        if imaging_type_pattern.match(f.stem) is None:
            print(f'A demo file: {str(f.name)}, skip.')
            continue

        name_info = f.stem.split('_')[0]
        imaging_type = name_info[0:2]
        person_id = int(name_info[2:])
        gt_path = str(f).replace(
            'images', 'ground_truth').replace('.tif', '.json')

        if person_id not in dataset:
            dataset[person_id] = []

        dataset[person_id].append({
            'img_path': str(f).replace(MIDV_ROOT, 'midv_dataset'),
            'gt_path': gt_path.replace(MIDV_ROOT, 'midv_dataset'),
            'imaging_type': imaging_type,
        })
    D.dump_json(dataset, DIR / 'midv_dataset.json')


def build_indoor_dataset():
    """
    Build a dataset for indoor scene recognition.

    This function reads indoor scene images from the INDOOR_ROOT/Images directory,
    processes them, and stores the resulting dataset in a JSON file named
    'indoor_background_dataset.json'.

    The indoor scene recognition dataset contains images in the .jpg format,
    organized into directories representing different indoor scenes.
    Each image belongs to a specific indoor scene category.

    The function performs the following steps:
        1. Read all .jpg image files from the INDOOR_ROOT/Images directory.
        2. Extract information about the scene category from the parent directory
            of each image.
        3. Create a list of dictionaries, with each dictionary containing the
            scene category and image path.
        4. Store the dataset in the 'indoor_background_dataset.json' file.

        Example of a dataset entry:
        {
            'scene': 'kitchen',
            'img_path': '/indoor_scene_recognition/Images/kitchen/image001.jpg'
        }
    """
    dataset = []
    indoor_fs = D.get_files(Path(INDOOR_ROOT) / 'Images', suffix=['.jpg'])
    for f in D.Tqdm(indoor_fs):
        dataset.append({
            'scene': f.parent.name,
            'img_path': str(f).replace(INDOOR_ROOT, 'indoor_scene_recognition'),
        })
    D.dump_json(dataset, DIR / 'indoor_dataset.json')


def build_cord_v0_dataset():
    for mode in ['train', 'test', 'dev']:
        dataset = []
        cord_fs = D.get_files(Path(CORD_ROOT) / mode, suffix=['.json'])
        for f in cord_fs:
            gt = D.load_json(f)

            if not gt['roi'] or len(gt['roi']) != 8:
                continue

            quad = [
                [gt['roi']['x1'], gt['roi']['y1']],
                [gt['roi']['x2'], gt['roi']['y2']],
                [gt['roi']['x3'], gt['roi']['y3']],
                [gt['roi']['x4'], gt['roi']['y4']]
            ]

            img_path = str(f.with_suffix('.png')).replace('json', 'image')
            if not Path(img_path).exists():
                continue

            dataset.append({
                'img_path': img_path.replace(CORD_ROOT, 'cord_v0'),
                'quad': quad,
            })

        D.dump_json(dataset, DIR / f'cord_v0_{mode}_dataset.json')


if __name__ == '__main__':
    build_midv_dataset()
    build_indoor_dataset()
    build_cord_v0_dataset()
