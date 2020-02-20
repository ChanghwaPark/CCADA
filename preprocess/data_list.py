import random

import numpy as np
import torch
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


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(Dataset):
    """
    Args:
        image_list (list): List of (image path, class_index) tuples
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
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


class CategoricalDataset(Dataset):
    def __init__(self, src_file, tgt_file, tgt_indices, tgt_predictions, selected_classes,
                 min_images_per_class=3, transform=None, target_transform=None, loader=default_loader):
        src_images_file_path_list = open(src_file).readlines()
        tgt_images_file_path_list = open(tgt_file).readlines()
        src_images = make_dataset(src_images_file_path_list, labels=None)
        tgt_images = make_dataset(tgt_images_file_path_list, labels=tgt_predictions)
        assert len(src_images) * len(tgt_images) > 0
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.selected_classes = selected_classes

        self.images = {'src': {}, 'tgt': {}}
        for selected_class in selected_classes:
            self.images['src'][selected_class] = [(image, label) for (image, label) in src_images
                                                  if label == selected_class]
            self.images['tgt'][selected_class] = [tgt_images[index] for index in tgt_indices
                                                  if tgt_images[index][1] == selected_class]
        self.batch_size_per_class = min_images_per_class

    def __getitem__(self, index):
        class_index = self.selected_classes[index]
        data = {}
        for domain in ['src', 'tgt']:
            domain_images = self.images[domain]

            image_indices = random.sample(range(len(domain_images[class_index])), self.batch_size_per_class)
            sampled_images = [domain_images[class_index][image_index] for image_index in image_indices]

            assert (len(sampled_images) == self.batch_size_per_class)
            for sampled_image in sampled_images:
                path, target = sampled_image
                assert target == class_index
                img = self.loader(path)

                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                if domain + '_images' not in data:
                    data[domain + '_images'] = [img]
                else:
                    data[domain + '_images'] += [img]

            data[domain + '_labels'] = [class_index] * len(data[domain + '_images'])
            data[domain + '_images'] = torch.stack(data[domain + '_images'], dim=0)

        return data

    def __len__(self):
        return len(self.selected_classes)
