import albumentations
import albumentations.augmentations.transforms
import albumentations.pytorch
from torch.utils.data import DataLoader

from .data_list import ImageList, ConfidentImageList, UniformImageList

# from torchvision import transforms

_RESIZE_SIZE = 256
_CROP_SIZE = 224
# _NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_NORMALIZE = albumentations.augmentations.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_data_loader(summary_file, data_loader_kwargs, training=True, is_center=False):
    transformer = get_transformer(training=training, is_center=is_center)
    dataset = ImageList(summary_file, transform=transformer)
    return DataLoader(dataset, **data_loader_kwargs)


def get_conf_data_loader(summary_file, data_loader_kwargs, conf_pair, min_conf_classes, is_center=False):
    transformer = get_transformer(training=True, is_center=is_center)
    dataset = ConfidentImageList(summary_file, conf_pair, min_conf_classes, transform=transformer)
    if dataset.conf_images is None:
        return None
    elif len(dataset.conf_images) < data_loader_kwargs['batch_size']:
        return None
    else:
        return DataLoader(dataset, **data_loader_kwargs)


# def get_certain_data_loader(summary_file, data_loader_kwargs, pseudo_labels, num_confident_samples=3, is_center=False):
#     transformer = get_transformer(training=True, is_center=is_center)
#     dataset = CustomImageList(summary_file, pseudo_labels, num_confident_samples=num_confident_samples,
#                               transform=transformer)
#     return DataLoader(dataset, **data_loader_kwargs)


def get_uniform_data_loader(summary_file, data_loader_kwargs, is_center=False):
    transformer = get_transformer(training=True, is_center=is_center)
    dataset = UniformImageList(summary_file, transform=transformer)
    return DataLoader(dataset, **data_loader_kwargs)


# def get_transformer(training, is_center=False):
#     if training:
#         if is_center:
#             crop_function = transforms.CenterCrop(_CROP_SIZE)
#         else:
#             crop_function = transforms.RandomResizedCrop(_CROP_SIZE)
#             # crop_function = transforms.RandomCrop(_CROP_SIZE)
#
#         return transforms.Compose([transforms.Resize(_RESIZE_SIZE),
#                                    crop_function,
#                                    transforms.RandomHorizontalFlip(),
#                                    transforms.ToTensor(),
#                                    _NORMALIZE])
#     else:
#         del is_center
#         return transforms.Compose([transforms.Resize(_RESIZE_SIZE),
#                                    transforms.CenterCrop(_CROP_SIZE),
#                                    transforms.ToTensor(),
#                                    _NORMALIZE])

# using albumentations
def get_transformer(training, is_center=False):
    if training:
        if is_center:
            crop_function = albumentations.CenterCrop(_CROP_SIZE, _CROP_SIZE)
        else:
            crop_function = albumentations.RandomResizedCrop(_CROP_SIZE, _CROP_SIZE)
            # crop_function = albumentations.RandomCrop(_CROP_SIZE, _CROP_SIZE) # TODO

        return albumentations.Compose([albumentations.Resize(_RESIZE_SIZE, _RESIZE_SIZE),
                                       crop_function,
                                       albumentations.HorizontalFlip(),
                                       _NORMALIZE,
                                       albumentations.pytorch.transforms.ToTensorV2()])
    else:
        del is_center
        return albumentations.Compose([albumentations.Resize(_RESIZE_SIZE, _RESIZE_SIZE),
                                       albumentations.CenterCrop(_CROP_SIZE, _CROP_SIZE),
                                       _NORMALIZE,
                                       albumentations.pytorch.transforms.ToTensorV2()])
