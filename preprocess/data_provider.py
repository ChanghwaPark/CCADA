from itertools import groupby

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchvision import transforms

from utils import get_confident_data_stat
from .data_list import ImageList, CategoricalDataset

_RESIZE_SIZE = 256
_CROP_SIZE = 224
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_data_loader(summary_file, data_loader_kwargs, training=True, is_center=False):
    images_file_path_list = open(summary_file).readlines()
    transformer = get_transformer(training=training, is_center=is_center)
    dataset = ImageList(images_file_path_list, transform=transformer)
    return DataLoader(dataset, **data_loader_kwargs)


def get_confident_data_loader(src_file, tgt_file, predictions_list, confidences_list,
                              threshold=0.95, num_confident_classes=10, min_images_per_class=3,
                              num_workers=4, is_center=False):
    # compute confidence for target images
    tgt_confident_indices, tgt_confident_predictions = \
        get_confident_data_stat(predictions_list, confidences_list, threshold=threshold)
    tgt_confident_class_dict = {value: len(list(group)) for value, group in groupby(sorted(tgt_confident_predictions))}

    # make class candidates that has more than 'min_images_per_class' images
    selected_classes = [key for key, value in tgt_confident_class_dict.items() if value >= min_images_per_class]
    if len(selected_classes) < num_confident_classes:
        return None
    transformer = get_transformer(training=True, is_center=is_center)
    confident_categorical_dataset = CategoricalDataset(src_file, tgt_file,
                                                       tgt_confident_indices, predictions_list, selected_classes,
                                                       min_images_per_class=min_images_per_class,
                                                       transform=transformer)
    sampler = RandomSampler(confident_categorical_dataset)
    batch_sampler = BatchSampler(sampler, num_confident_classes, drop_last=True)

    # confident_categorical_loader = DataLoader(confident_categorical_dataset,
    #                                           collate_fn=collate_fn,
    #                                           shuffle=True,
    #                                           drop_last=True,
    #                                           batch_size=num_confident_classes,
    #                                           num_workers=num_workers)
    confident_categorical_loader = DataLoader(confident_categorical_dataset,
                                              batch_sampler=batch_sampler,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)

    return confident_categorical_loader


def get_transformer(training, is_center=False):
    if training:
        if is_center:
            crop_function = transforms.CenterCrop(_CROP_SIZE)
        else:
            crop_function = transforms.RandomResizedCrop(_CROP_SIZE)
            # crop_function = transforms.RandomCrop(_CROP_SIZE) # TODO

        return transforms.Compose([transforms.Resize(_RESIZE_SIZE),
                                   crop_function,
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   _NORMALIZE])
    else:
        del is_center
        return transforms.Compose([transforms.Resize(_RESIZE_SIZE),
                                   transforms.CenterCrop(_CROP_SIZE),
                                   transforms.ToTensor(),
                                   _NORMALIZE])


def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('labels') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('images') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate
