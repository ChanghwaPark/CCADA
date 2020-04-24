import albumentations
import albumentations.augmentations.transforms
import albumentations.pytorch
from torch.utils.data import DataLoader

from .confident_dataset import ConfidentDataset
from .default_dataset import DefaultDataset
from .uniform_dataset import UniformDataset

_RESIZE_SIZE = 256
_CROP_SIZE = 224
_NORMALIZE = albumentations.augmentations.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# using albumentations
def get_transform(training):
    if training:
        crop_function = albumentations.RandomResizedCrop(_CROP_SIZE, _CROP_SIZE)  # TODO
        # crop_function = albumentations.RandomCrop(_CROP_SIZE, _CROP_SIZE)
        # crop_function = albumentations.CenterCrop(_CROP_SIZE, _CROP_SIZE)

        return albumentations.Compose([albumentations.Resize(_RESIZE_SIZE, _RESIZE_SIZE),
                                       crop_function,
                                       albumentations.HorizontalFlip(),
                                       _NORMALIZE,
                                       albumentations.pytorch.transforms.ToTensorV2()])
    else:
        return albumentations.Compose([albumentations.Resize(_RESIZE_SIZE, _RESIZE_SIZE),
                                       albumentations.CenterCrop(_CROP_SIZE, _CROP_SIZE),
                                       _NORMALIZE,
                                       albumentations.pytorch.transforms.ToTensorV2()])


class BaseDataLoader(object):
    def __init__(self, summary_file, data_loader_kwargs, training=True):
        self.summary_file = summary_file
        self.data_loader_kwargs = data_loader_kwargs
        self.transformer = get_transform(training=training)
        self.dataset = None
        self.data_loader = None

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def construct_data_loader(self):
        raise NotImplementedError


class DefaultDataLoader(BaseDataLoader):
    def __init__(self, summary_file, data_loader_kwargs, training=True):
        super().__init__(summary_file, data_loader_kwargs, training=training)
        self.construct_data_loader()

    def construct_data_loader(self):
        self.dataset = DefaultDataset(self.summary_file, transform=self.transformer)
        self.data_loader = DataLoader(self.dataset, **self.data_loader_kwargs)


class UniformDataLoader(BaseDataLoader):
    def __init__(self, summary_file, data_loader_kwargs, training=True):
        super().__init__(summary_file, data_loader_kwargs, training=training)
        self.construct_data_loader()

    def construct_data_loader(self):
        self.dataset = UniformDataset(self.summary_file, transform=self.transformer)
        self.data_loader = DataLoader(self.dataset, **self.data_loader_kwargs)


class ConfidentDataLoader(BaseDataLoader):
    def __init__(self, summary_file, data_loader_kwargs, conf_pair, min_conf_classes, training=True):
        super().__init__(summary_file, data_loader_kwargs, training=training)
        self.conf_pair = conf_pair
        self.min_conf_classes = min_conf_classes
        self.construct_data_loader()

    def construct_data_loader(self):
        self.dataset = ConfidentDataset(self.summary_file, self.conf_pair, self.min_conf_classes,
                                        transform=self.transformer)
        if self.dataset.conf_images is None:
            self.data_loader = None
        elif len(self.dataset.conf_images) < self.data_loader_kwargs['batch_size']:
            self.data_loader = None
        else:
            self.data_loader = DataLoader(self.dataset, **self.data_loader_kwargs)
