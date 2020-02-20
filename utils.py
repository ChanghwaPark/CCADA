import queue

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
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


def summary_write_figures(summary_writer, tag, global_step, model, images, labels, domain):
    model.set_bn_domain(domain=domain)
    model.eval()
    with torch.no_grad():
        # _, predictions, confidences = model.images_to_probabilities(images)
        end_points = model(images)
        # figure = plot_classes_predictions(images, labels, predictions, confidences)
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


def summary_write_embeddings(summary_writer, tag, global_step, model, src_train_loader, tgt_train_loader,
                             num_samples=128):
    total_iteration = num_samples // src_train_loader.batch_size
    model.eval()
    with torch.no_grad():
        features_list = []
        class_labels_list = []
        domain_labels_list = []
        # for ((src_inputs, src_labels), (tgt_inputs, tgt_labels)) in zip(src_train_loader, tgt_train_loader):
        for ((src_inputs, src_labels, _), (tgt_inputs, tgt_labels, _)) in zip(src_train_loader, tgt_train_loader):
            src_inputs, tgt_inputs, src_labels, tgt_labels = \
                src_inputs.cuda(), tgt_inputs.cuda(), src_labels.cuda(), tgt_labels.cuda()
            # inputs = torch.cat((src_inputs, tgt_inputs), dim=0)
            model.set_bn_domain(domain=0)
            src_end_points = model(src_inputs)
            model.set_bn_domain(domain=1)
            tgt_end_points = model(tgt_inputs)
            # end_points = model(inputs)
            # features = end_points[tag]
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


def summary_write_images(summary_writer, tag, global_step, images):
    images = inv_normalize_images(images)
    img_grid = torchvision.utils.make_grid(images, nrow=4)
    summary_writer.add_image(tag=tag,
                             img_tensor=img_grid,
                             global_step=global_step)
    summary_writer.close()


def inv_normalize_images(images, batch_size=32):
    images = ImageTransform(images, transform=inv_normalize)
    image_loader = DataLoader(images, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=0)
    images_list = []
    for transformed_images in image_loader:
        images_list.append(transformed_images)
    all_transformed_images = torch.cat(images_list, dim=0)
    return all_transformed_images


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


def get_confident_data_stat(predictions_list, confidences_list, threshold=0.95):
    confident_data_indices = [index for index, confidence in enumerate(confidences_list) if confidence > threshold]
    confident_data_predictions = [predictions_list[index] for index in confident_data_indices]

    return confident_data_indices, confident_data_predictions


def get_dataset_name(src_name, tgt_name):
    # only office dataset is implemented. other datasets should be added later
    dataset_names = {
        'amazon': 'office',
        'dslr': 'office',
        'webcam': 'office'
    }
    assert (dataset_names[src_name] == dataset_names[tgt_name])
    return dataset_names[src_name]


def prepare_categorical_data(categorical_data):
    src_inputs = torch.cat([class_images.cuda() for class_images in categorical_data['src_images']], dim=0)
    src_labels = torch.cat([class_labels.cuda() for class_labels in categorical_data['src_labels']], dim=0)
    tgt_inputs = torch.cat([class_images.cuda() for class_images in categorical_data['tgt_images']], dim=0)
    tgt_labels = torch.cat([class_labels.cuda() for class_labels in categorical_data['tgt_labels']], dim=0)
    return src_inputs, src_labels, tgt_inputs, tgt_labels


class AverageMeter:
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


def get_n_dims(local_feature_name):
    name_to_n_dims = {
        'r1': 1,
        'r7': 7,
        'r14': 14,
        'r28': 28
    }
    return name_to_n_dims[local_feature_name]


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)
