import os
import shutil
import glob
import argparse

import random
import yaml
import tqdm
import numpy as np
import cv2


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


def compute_anchors(angle, bin=2, overlap=0.1):
    """Compute anchors

    Args
    :angle:
    
    Returns
    :anchors:
    """
    anchors = []
    
    wedge = 2.*np.pi / bin
    l_index = int(angle/wedge)
    r_index = l_index + 1
    
    if (angle - l_index*wedge) < wedge/2 * (1 + overlap / 2):
        anchors.append([l_index, angle - l_index*wedge])
        
    if (r_index*wedge - angle) < wedge/2 * (1 + overlap / 2):
        anchors.append([r_index % bin, angle - r_index*wedge])
        
    return anchors


def load_data(image_dir, label_dir, class_names, bin=2):
    """Load KITTI data for writting training data in YOLO format.

    
    Args
    :image_dir:
    :label_dir:
    
    Returns
    """
    all_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in class_names}
    dims_cnt = {key:0 for key in class_names}
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for label_file in sorted(os.listdir(label_dir)):
        image_file = label_file.replace('txt', 'png')

        for line in open(os.path.join(label_dir, label_file)).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in class_names and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name': line[0],
                       'class_id': class_map[line[0]],
                       'image': os.path.join(image_dir, image_file),
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)

    data = dict()
    for obj in all_objs:
        # Fix dimensions
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((bin, 2))
        confidence = np.zeros(bin)

        anchors = compute_anchors(obj['new_alpha'])

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Group by file name
        filename = obj['image']
        if filename not in data:
            data[filename] = []
        
        del obj['image']
        data[filename].append(obj)

    return data


def write_data(data, output_dir, split):
    """
    """
    image_dir = os.path.join(output_dir, 'images', split)
    os.makedirs(image_dir, exist_ok=True)

    label_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(label_dir, exist_ok=True)
    for image_path, objs in tqdm.tqdm(data.items()):
        image_filename = os.path.basename(image_path)
        new_image_path = os.path.join(image_dir, image_filename)
        label_filename = image_filename.replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'wt') as f:
            for obj in objs:

                class_id = obj['class_id']
                image = cv2.imread(image_path)
                h, w = image.shape[:2]
                xmin, ymin = obj['xmin'], obj['ymin']
                xmax, ymax = obj['xmax'], obj['ymax']
                xmin, ymin = xmin / w, ymin / h
                xmax, ymax = xmax / w, ymax / h

                dims = obj['dims']
                orient = obj['orient']
                conf = obj['conf']

                orient = [x for row in orient for x in row]
                row = [class_id, xmin, ymin, xmax, ymax, *dims, *orient, *conf]
                f.write(' '.join(map(str, row)) + '\n')
        
        shutil.copy(image_path, new_image_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/kitti/training/image_2', help='Path to kitti image dir')
    parser.add_argument('--output_dir', type=str, default='./data/kitti', help='Output dir')
    parser.add_argument('--data', type=str, default='data/kitti.yaml', help='data.yaml path')
    args = parser.parse_args()
    print(args)

    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    
    if not os.path.isdir(args.image_dir):
        raise Exception('Not found training directory: {}'.format(args.train_dir))

    image_dir = args.image_dir
    label_dir = image_dir.replace('image_2', 'label_2')
    class_names = data_dict['names']

    # Load data
    print('Loading data...')
    data = load_data(image_dir, label_dir, class_names)
    split = 0.1
    num_train = int(len(data) * (1 - split))
    random.seed(42)
    items = list(data.items())
    random.shuffle(items)
    train_data = dict(items[:num_train])
    val_data = dict(items[num_train:])
    print('Number of training samples: {}'.format(len(train_data)))
    print('Number of validation samples: {}'.format(len(val_data)))

    # Write data in YOLO format
    print('Creating training set...')
    write_data(train_data, args.output_dir, 'train')

    print('Creating validatoin set...')
    write_data(val_data, args.output_dir, 'val')


if __name__ == '__main__':
    main()
