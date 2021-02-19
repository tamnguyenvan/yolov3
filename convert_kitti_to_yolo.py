import os
import shutil
import glob
import argparse

import tqdm
import numpy as np
import cv2


KITTI_NAMES = [
    'Person_sitting', 'Pedestrian',
    'Car', 'Tram', 'Misc', 'Cyclist', 'Truck', 'Van'
]


def load_txt(text_file, img_width, img_height):
    """
    """
    all_data = []
    with open(text_file) as f:
        for line in f:
            data = line.strip().split()
            if len(data) == 15:
                class_str = data[0]
                if class_str != 'DontCare':
                    x1 = float(data[4])
                    y1 = float(data[5])
                    x2 = float(data[6])
                    y2 = float(data[7])
                    
                    bbox_center_x = float( (x1 + (x2 - x1) / 2.0) / img_width)
                    bbox_center_y = float( (y1 + (y2 - y1) / 2.0) / img_height)
                    bbox_width = float((x2 - x1) / img_width)
                    bbox_height = float((y2 - y1) / img_height)
                    all_data.append((class_str, bbox_center_x, bbox_center_y, bbox_width, bbox_height))
    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/kitti/training', help='Path to kitti image dir')
    parser.add_argument('--output_dir', type=str, default='./data/kitti', help='Output dir')
    args = parser.parse_args()
    
    # Load image paths and label paths
    if not os.path.isdir(args.train_dir):
        raise Exception('Not found training directory: {}'.format(args.train_dir))

    img_paths = glob.glob(os.path.join(args.train_dir, 'image_2', '*.png'))
    label_paths = []
    for img_path in img_paths:
        img_filename = os.path.basename(img_path)
        img_filename = os.path.splitext(img_filename)[0]
        label_path = os.path.join(args.train_dir, 'label_2', img_filename + '.txt')
        label_paths.append(label_path)

    # Split
    val_split = 0.1
    test_split = 0.1
    num_val = int(len(img_paths) * val_split)
    num_test = int(len(img_paths) * test_split)
    num_train = len(img_paths) - num_val - num_test
    np.random.seed(42)
    indices = np.arange(len(img_paths))
    np.random.shuffle(indices)

    img_paths = np.array(img_paths)[indices].tolist()
    label_paths = np.array(label_paths)[indices].tolist()

    train_img_paths = img_paths[:num_train]
    train_label_paths = label_paths[:num_train]

    val_img_paths = img_paths[num_train:num_train+num_val]
    val_label_paths = label_paths[num_train:num_train+num_val]

    test_img_paths = img_paths[num_train+num_val:]
    test_label_paths = label_paths[num_train+num_val:]

    print('Number of training: {}'.format(len(train_img_paths)))
    print('Number of validation: {}'.format(len(val_img_paths)))
    print('Number of test: {}'.format(len(test_img_paths)))

    # Prepare outputs
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    image_dir = os.path.join(out_dir, 'images')
    label_dir = os.path.join(out_dir, 'labels')
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(image_dir, split), exist_ok=True)
        os.makedirs(os.path.join(label_dir, split), exist_ok=True)
    
    # Load kitti class names
    sources = {
        'train': (train_img_paths, train_label_paths),
        'val': (val_img_paths, val_label_paths),
        'test': (test_img_paths, test_label_paths)
    }
    class_map = {class_name: idx for idx, class_name in enumerate(KITTI_NAMES)}
    for split, data_source in sources.items():
        print('Creating {}'.format(split))
        split_img_dir = os.path.join(image_dir, split)
        split_label_dir = os.path.join(label_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)
        for img_path, label_path in tqdm.tqdm(zip(*data_source)):
            img_filename = os.path.basename(img_path)
            dst_img_path = os.path.join(split_img_dir, img_filename)
            shutil.copy(img_path, dst_img_path)

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            label_filename = os.path.basename(label_path)
            dst_label_path = os.path.join(split_label_dir, label_filename)
            label_all_data = load_txt(label_path, w, h)
            with open(dst_label_path, 'wt') as f:
                for label_data in label_all_data:
                    class_name = label_data[0]
                    bbox_data = label_data[1:]
                    class_id = class_map[class_name]
                    dump_data = [class_id, *bbox_data]
                    f.write(' '.join(map(str, dump_data)) + '\n')


if __name__ == '__main__':
    main()
