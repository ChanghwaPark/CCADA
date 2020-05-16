from .base_dataset import BaseDataset
from .utils import cv_loader


class IndicesDataset(BaseDataset):
    def __init__(self, summary_file, indices,
                 labels=None, transform=None, target_transform=None, loader=cv_loader):
        super().__init__(summary_file,
                         labels=labels, transform=transform, target_transform=target_transform, loader=loader)
        self.indices_images = [self.entire_images[index] for index in indices]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: keys are (image, pseudo_label)
        """
        data = self.get_image_and_label(self.indices_images, index)

        return data

    def __len__(self):
        return len(self.indices_images)
