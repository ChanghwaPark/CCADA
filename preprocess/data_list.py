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

    # def __init__(self, image_list, labels=None, transform=None, target_transform=None,
    #              loader=default_loader):
    def __init__(self, summary_file, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
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
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target, index

    def __len__(self):
        return len(self.images)


class CustomImageList(Dataset):
    def __init__(self, summary_file, pseudo_labels, min_dataset_size=360, transform=None, loader=default_loader):
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
        shuffle(confident_classes)

        sorted_confident_images = {}
        for confident_class in confident_classes:
            sorted_confident_images[confident_class] \
                = [(path, ori_index) for (path, ori_index, pseudo_label)
                   in confident_images if pseudo_label == confident_class]
            # initial shuffle
            shuffle(sorted_confident_images[confident_class])

        temp_confident_images = copy.deepcopy(sorted_confident_images)
        num_samples = 0
        rearranged_images = []
        while num_samples < min_dataset_size:
            for confident_class in confident_classes:
                if len(temp_confident_images[confident_class]) == 0:
                    temp_confident_images[confident_class] = copy.deepcopy(sorted_confident_images[confident_class])
                    shuffle(temp_confident_images[confident_class])
                rearranged_images.append(temp_confident_images[confident_class].pop())
                num_samples += 1

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
