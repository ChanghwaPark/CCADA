from .base_dataset import BaseDataset
from .utils import cv_loader, class_uniform_rearranger


class UniformDataset(BaseDataset):
    def __init__(self, summary_file,
                 labels=None, transform=None, target_transform=None, loader=cv_loader):
        super().__init__(summary_file,
                         labels=labels, transform=transform, target_transform=target_transform, loader=loader)
        rearranged_images = class_uniform_rearranger(self.entire_images)
        self.rearranged_images = rearranged_images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: keys are (image, true_label)
        """
        data = self.get_image_and_label(self.rearranged_images, index)

        return data

    def __len__(self):
        return len(self.rearranged_images)
