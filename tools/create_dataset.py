import os
import json
from PIL import Image
from shapely import Polygon
import numpy as np
import pycocotools.mask as maskUtils
import cv2


def binary_mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for _, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def binary_mask_to_rle2(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(
        groupby(binary_mask.ravel(order="F"))
    ):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


if __name__ == "__main__":
    image_root = "D:/HocTap/KLTN/dataset/processed/mask_create/300"
    split_percents = [0.3, 0.5, 0.7]

    total = 0
    for image_name in os.listdir(image_root):
        if "mask" in image_name:
            continue

        image_path = f"{image_root}/{image_name}"
        mask_name = f"{image_name.split('.')[0]}_mask.jpg"
        mask_path = f"{image_root}/{mask_name}"

        assert os.path.exists(mask_path), f"Cannot find {mask_path=}"

        mask = np.array(Image.open(mask_path))

        # rle = maskUtils.encode(np.asfortranarray(mask))
        # rle_mask = maskUtils.decode(rle).squeeze()

    print(total / len(os.listdir(image_root)))
