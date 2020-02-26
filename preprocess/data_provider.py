from torch.utils.data import DataLoader
from torchvision import transforms

from .data_list import ImageList, CustomImageList

_RESIZE_SIZE = 256
_CROP_SIZE = 224
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_data_loader(summary_file, data_loader_kwargs, training=True, is_center=False):
    transformer = get_transformer(training=training, is_center=is_center)
    dataset = ImageList(summary_file, transform=transformer)
    return DataLoader(dataset, **data_loader_kwargs)


def get_certain_data_loader(summary_file, data_loader_kwargs, confident_mask, is_center=False):
    transformer = get_transformer(training=True, is_center=is_center)
    dataset = CustomImageList(summary_file, confident_mask, transform=transformer)
    return DataLoader(dataset, **data_loader_kwargs)


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
