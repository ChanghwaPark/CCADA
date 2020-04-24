from .base_dataset import BaseDataset
from .utils import cv_loader


class DefaultDataset(BaseDataset):
    def __init__(self, summary_file,
                 labels=None, transform=None, target_transform=None, loader=cv_loader):
        super().__init__(summary_file,
                         labels=labels, transform=transform, target_transform=target_transform, loader=loader)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: keys are (image, true_label, index) where index is sample index of the entire dataset.
        """
        data = self.get_image_and_label(self.entire_images, index)
        data['index'] = index

        return data

    def __len__(self):
        return len(self.entire_images)
