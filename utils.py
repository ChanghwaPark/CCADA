import queue

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def configure(filename):
    with open(filename, 'r') as f:
        parser = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser


def summary_write_fig(summary_writer, tag, global_step, model, images, labels, domain):
    model.set_bn_domain(domain=domain)
    model.eval()

    with torch.no_grad():
        end_points = model(images)
        figure = plot_classes_predictions(images, labels, end_points['predictions'], end_points['confidences'])

    summary_writer.add_figure(tag=tag,
                              figure=figure,
                              global_step=global_step)
    summary_writer.close()


def plot_classes_predictions(images, labels, predictions, confidences):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    """
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 30))
    for idx in np.arange(min(32, len(images))):
        ax = fig.add_subplot(8, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            predictions[idx],
            confidences[idx] * 100.0,
            labels[idx]),
            color=("green" if predictions[idx] == labels[idx] else "red"))
    return fig


def matplotlib_imshow(image):
    np_image = image.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = np_image * np.array(_STD) + np.array(_MEAN)
    np_image = np.clip(np_image, 0., 1.)
    plt.imshow(np_image)


def summary_write_proj(summary_writer, tag, global_step, model, src_train_loader, tgt_train_loader,
                       num_samples=128):
    total_iteration = num_samples // src_train_loader.data_loader.batch_size
    model.eval()
    with torch.no_grad():
        features_list = []
        class_labels_list = []
        domain_labels_list = []

        for (src_data, tgt_data) in zip(src_train_loader, tgt_train_loader):
            src_inputs, src_labels = src_data['image'].cuda(), src_data['true_label'].cuda()
            tgt_inputs, tgt_labels = tgt_data['image'].cuda(), tgt_data['true_label'].cuda()
            model.set_bn_domain(domain=0)
            src_end_points = model(src_inputs)
            model.set_bn_domain(domain=1)
            tgt_end_points = model(tgt_inputs)
            src_features = src_end_points[tag]
            tgt_features = tgt_end_points[tag]
            features = torch.cat([src_features, tgt_features], dim=0)
            features_list.append(features)

            class_labels = torch.cat((src_labels, tgt_labels), dim=0)
            class_labels_list.append(class_labels)

            domain_labels = ['S'] * src_labels.size(0) + ['T'] * tgt_labels.size(0)
            domain_labels_list.extend(domain_labels)

            if len(features_list) >= total_iteration:
                break

        all_features = torch.cat(features_list, dim=0)
        all_class_labels = torch.cat(class_labels_list, dim=0)
        all_class_labels = all_class_labels.cpu().numpy()

    summary_writer.add_embedding(all_features,
                                 metadata=all_class_labels,
                                 global_step=global_step,
                                 tag=tag + "_class")
    summary_writer.add_embedding(all_features,
                                 metadata=domain_labels_list,
                                 global_step=global_step,
                                 tag=tag + "_domain")
    summary_writer.close()


class ImageTransform(Dataset):
    def __init__(self, images, transform=None):
        assert len(images) > 0
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


def get_dataset_name(src_name, tgt_name):
    dataset_names = {
        'amazon': 'office',
        'dslr': 'office',
        'webcam': 'office',
        'c': 'image-clef',
        'i': 'image-clef',
        'p': 'image-clef',
        'art': 'office-home',
        'clipart': 'office-home',
        'product': 'office-home',
        'real_world': 'office-home',
        'visda_src': 'visda',
        'visda_tgt': 'visda'
    }
    assert (dataset_names[src_name] == dataset_names[tgt_name])
    return dataset_names[src_name]


class AvgMeter:
    def __init__(self, maxsize=10):
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, item):
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)

    def get(self):
        return self.queue.get()

    def get_average(self):
        sum_all = 0.
        queue_len = self.queue.qsize()
        if queue_len == 0:
            return 0
        while not self.queue.empty():
            sum_all += self.queue.get()
        return round(sum_all / queue_len, 5)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_labels_from_file(file_name):
    image_list = open(file_name).readlines()
    labels = [int(val.split()[1]) for val in image_list]
    return labels


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def compute_accuracy(logits, true_labels, acc_metric='total_mean', print_result=False):
    assert logits.size(0) == true_labels.size(0)
    if acc_metric == 'total_mean':
        predictions = torch.max(logits, dim=1)[1]
        return 100.0 * (predictions == true_labels).sum().item() / logits.size(0)
    elif acc_metric == 'class_mean':
        num_classes = logits.size(1)
        predictions = torch.max(logits, dim=1)[1]
        class_accuracies = []
        for class_label in range(num_classes):
            class_mask = (true_labels == class_label)

            class_count = class_mask.sum().item()
            if class_count == 0:
                class_accuracies += [0.0]
                continue

            class_accuracy = 100.0 * (predictions[class_mask] == class_label).sum().item() / class_count
            class_accuracies += [class_accuracy]
        if print_result:
            print(f'class_accuracies: {class_accuracies}')
        return np.mean(class_accuracies)
    else:
        raise ValueError(f'acc_metric, {acc_metric} is not available.')
