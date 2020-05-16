from torch.utils.data import Dataset

from .utils import make_dataset


class BaseDataset(Dataset):
    """
    Args:
        summary_file (file_path): Path to images summary file
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(self, summary_file, labels=None, transform=None, target_transform=None, loader=None):
        image_list = open(summary_file).readlines()
        entire_images = make_dataset(image_list, labels)
        assert len(entire_images) > 0

        self.entire_images = entire_images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def get_image_and_label(self, images, index, label_name='true_label'):
        data = {}

        path, label = images[index]
        image = self.loader(path)
        if self.transform is not None:
            image_1 = self.transform(image=image)
            image_1 = image_1['image']
            image_2 = self.transform(image=image)
            image_2 = image_2['image']
        else:
            image_1 = image
            image_2 = image
        data['image_1'] = image_1
        data['image_2'] = image_2

        if self.target_transform is not None:
            label = self.target_transform(label)
        data[label_name] = label

        return data
