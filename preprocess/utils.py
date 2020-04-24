import random

import cv2
import numpy as np


def get_classes_set(images):
    """
    From images get unique class set
    Args:
        images (path, target): Images to be processed.
    """
    targets = [target for (_, target) in images]
    target_classes = list(set(targets))
    # random.shuffle(target_classes)
    return target_classes


def class_uniform_rearranger(images):
    """
    Args:
        images (path, target): Images to be rearranged.
    """
    target_classes = get_classes_set(images)
    # random.shuffle(target_classes)

    sorted_images = {}
    rearranged_images = []
    min_class_len = len(images)
    print(f'len(target_classes): {len(target_classes)}')

    for target_class in target_classes:
        sorted_images[target_class] = [(path, target) for (path, target) in images if target == target_class]
        random.shuffle(sorted_images[target_class])
        min_class_len = min(min_class_len, len(sorted_images[target_class]))
        print(f'class {target_class}: {len(sorted_images[target_class])}')

    print(f'min_class_len: {min_class_len}')

    for _ in range(min_class_len):
        for target_class in target_classes:
            rearranged_images.append(sorted_images[target_class].pop())

    print(f'len(rearranged_images): {len(rearranged_images)}')

    return rearranged_images


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        # images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
        images = [(image_list[i].split()[0], labels[i]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def cv_loader(path):
    img = cv2.imread(path)
    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
