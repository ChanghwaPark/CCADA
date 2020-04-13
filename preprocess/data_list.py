import copy
from random import shuffle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(Dataset):
    """
    Args:
        summary_file (file_path): Path to images summary file
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(self, summary_file, labels=None, transform=None, target_transform=None, loader=default_loader):
        image_list = open(summary_file).readlines()
        images = make_dataset(image_list, labels)
        assert len(images) > 0
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.images)


class CustomImageList(Dataset):
    def __init__(self, summary_file, pseudo_labels, num_confident_samples=4, transform=None, loader=default_loader):
        image_list = open(summary_file).readlines()
        images = make_dataset(image_list, None)
        assert len(images) > 0

        pseudo_labels_list = pseudo_labels.cpu().tolist()
        assert len(images) == len(pseudo_labels_list)

        confident_images = [(path, ori_index, pseudo_label) for ori_index, ((path, target), pseudo_label) in
                            enumerate(zip(images, pseudo_labels_list)) if pseudo_label >= 0]

        confident_classes = list(set(pseudo_labels_list))
        if -1 in confident_classes:
            confident_classes.remove(-1)
        temp_confident_classes = copy.deepcopy(confident_classes)
        for confident_class in temp_confident_classes:
            if pseudo_labels_list.count(confident_class) < num_confident_samples:
                confident_classes.remove(confident_class)
        shuffle(confident_classes)

        sorted_confident_images = {}
        min_class_len = len(confident_images)
        for confident_class in confident_classes:
            sorted_confident_images[confident_class] = [(path, ori_index) for (path, ori_index, pseudo_label)
                                                        in confident_images if pseudo_label == confident_class]
            shuffle(sorted_confident_images[confident_class])
            if min_class_len > len(sorted_confident_images[confident_class]):
                min_class_len = len(sorted_confident_images[confident_class])
        temp_confident_images = copy.deepcopy(sorted_confident_images)

        rearranged_images = []
        for _ in range(min_class_len):
            for confident_class in confident_classes:
                assert len(temp_confident_images[confident_class]) > 0
                rearranged_images.append(temp_confident_images[confident_class].pop())

        self.rearranged_images = rearranged_images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, ori_index = self.rearranged_images[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, ori_index

    def __len__(self):
        return len(self.rearranged_images)


class UniformImageList(Dataset):
    def __init__(self, summary_file, labels=None, transform=None, target_transform=None, loader=default_loader):
        image_list = open(summary_file).readlines()
        images = make_dataset(image_list, labels)
        assert len(images) > 0

        targets = [target for (_, target) in images]
        target_classes = list(set(targets))
        shuffle(target_classes)

        sorted_images = {}
        min_class_len = len(images)
        for target_class in target_classes:
            sorted_images[target_class] = [(path, ori_index, target) for ori_index, (path, target) in
                                           enumerate(images) if target == target_class]
            shuffle(sorted_images[target_class])
            if min_class_len > len(sorted_images[target_class]):
                min_class_len = len(sorted_images[target_class])

        temp_sorted_images = copy.deepcopy(sorted_images)
        rearranged_images = []
        for _ in range(min_class_len):
            for target_class in target_classes:
                assert len(temp_sorted_images[target_class]) > 0
                rearranged_images.append(temp_sorted_images[target_class].pop())

        self.images = rearranged_images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, ori_index, target = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, ori_index

    def __len__(self):
        return len(self.images)
