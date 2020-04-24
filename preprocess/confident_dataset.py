from .base_dataset import BaseDataset
from .utils import cv_loader, class_uniform_rearranger, get_classes_set


class ConfidentDataset(BaseDataset):
    def __init__(self, summary_file, conf_pair, min_conf_classes,
                 labels=None, transform=None, target_transform=None, loader=cv_loader):
        super().__init__(summary_file,
                         labels=labels, transform=transform, target_transform=target_transform, loader=loader)

        conf_images = [self.entire_images[index] for (index, _) in conf_pair]
        conf_pseudo_labels = [pseudo_label for (_, pseudo_label) in conf_pair]

        # change true labels to the pseudo labels
        conf_images = [(path, pseudo_label) for ((path, _), pseudo_label) in zip(conf_images, conf_pseudo_labels)]

        # check if the pseudo labels set have min_conf_classes at least.
        conf_classes = get_classes_set(conf_images)
        if len(conf_classes) < min_conf_classes:
            print(f'len(conf_classes), {len(conf_classes)} < min_conf_classes, {min_conf_classes}')
            self.conf_images = None
        else:
            # rearrange confident images to be class-wisely uniform
            self.conf_images = class_uniform_rearranger(conf_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: keys are (image, pseudo_label)
        """
        data = self.get_image_and_label(self.conf_images, index, label_name='pseudo_label')

        return data

    def __len__(self):
        return len(self.conf_images)
