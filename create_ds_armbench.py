#!/usr/bin/env python3

"""Generate the ArmBench dataset 
"""

import argparse
import json
import re
import sys
import uuid
from PIL import Image
import h5py
import numpy as np
import os
from tqdm import tqdm


def save_dataset(h5_path, output_dir, train_size, val_size, test_size):
    with h5py.File(h5_path) as f:
        img = f["data"]
        mask = f["mask"]

        # get train_size number of images for training with single object
        # i = 0
        # train_img, train_mask, i = get_split(train_size, img, mask, i)
        # val_img, val_mask, i = get_split(val_size, img, mask, i)
        # test_img, test_mask, i = get_split(test_size, img, mask, i)

        img_dir = f"{output_dir}/images"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        train_dir = f"{output_dir}/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        val_dir = f"{output_dir}/validation"
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        test_dir = f"{output_dir}/test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        prep_and_save(img, mask, train_dir, img_dir, train_size, 0)
        prep_and_save(img, mask, val_dir, img_dir, val_size, train_size)
        prep_and_save(img, mask, test_dir, img_dir, test_size, train_size + val_size)



def prep_and_save(img, mask, json_dir, img_dir, size, start):
    json_data_list = []

    for i in tqdm(range(start, start + size), desc=json_dir):
        img_name = f"{i}.jpg"
        img_path = os.path.join(img_dir, img_name)
        img_i = img[i]
        img_i = img_i.astype(np.uint8)
        img_i = Image.fromarray(img_i)
        img_i.save(img_path)

        boxes = []
        for cat in np.unique(mask[i]):
            if cat == 0 or cat == 255:
                continue
            mask_i = mask[i]
            mask_i = mask_i == cat
            bbox = [
                int(np.min(np.where(mask_i)[1])),  # x_min
                int(np.min(np.where(mask_i)[0])),  # y_min
                int(np.max(np.where(mask_i)[1]))
                - int(np.min(np.where(mask_i)[1])),  # width
                int(np.max(np.where(mask_i)[0]))
                - int(np.min(np.where(mask_i)[0])),  # height
            ]
            bbox = bbox / np.array([512, 384, 512, 384])
            # round to 2 decimal places
            bbox = np.round(bbox, 2)
            boxes.append(bbox.tolist())

        if not boxes:
            continue

        json_data = {
            "id": i,
            "image": img_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "Perform object detection on the given image. First, resize the image to 1x1, then answer: what are the bounding boxes of the objects in the image? Output should be in  the format of a list of lists, where each list represents a bounding box: [[x_min, y_min, width, height], [x_min, y_min, width, height], ...] \n\nx_min and y_min are placeholder for the coordinate of the top left corner of an object bounding box, width and height are placeholder for the width and height of the bounding box. \n\nThe x_min, y_min, width and height of the bounding box should be positive decimal numbers between 0 and 1. \n\n \n\nFor example, if there are two objects in the image, the bounding boxes of the objects could be [[0.13, 0.24, 0.37, 0.4], [0.5, 0.55, 0.23, 0.25]].",
                },
                {"from": "gpt", "value": json.dumps(boxes)},
            ],
        }

        json_data_list.append(json_data)

    json_output_path = os.path.join(json_dir, "dataset.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data_list, json_file, indent=4)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--h5",
        type=str,
        help="h5 file containing the dataset",
        default="/gscratch/sciencehub/sebgab/Dev/data/armbench512x384_5_all.h5",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the dataset in",
        default="armbench",
        required=False,
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="Number of training samples",
        default=60000,
        required=False,
    )
    parser.add_argument(
        "--val_size",
        type=int,
        help="Number of validation samples",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Number of test samples",
        default=100,
        required=False,
    )
    args = parser.parse_args(arguments)
    h5_path = args.h5
    output_dir = args.output_dir
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    save_dataset(h5_path, output_dir, train_size, val_size, test_size)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
