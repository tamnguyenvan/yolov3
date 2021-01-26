#!/bin/bash
DATA_DIR=$(pwd)/../kitti
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Downloading kitti..."
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

echo "Extracting..."
unzip data_object_image_2.zip
unzip data_object_label_2.zip
echo "Done"
