import argparse
import os
import pickle

import torch
import tqdm
from termcolor import colored
from torch.utils.data import DataLoader

from model.model import Model
from preprocess.data_provider import get_transform
from preprocess.default_dataset import DefaultDataset
from utils import configure, get_dataset_name

parser = argparse.ArgumentParser()

# dataset configurations
parser.add_argument('--config',
                    type=str,
                    default='config/config.yml',
                    help='Dataset configuration parameters')
parser.add_argument('--dataset_root',
                    type=str,
                    default='/home/omega/datasets')
parser.add_argument('--src',
                    type=str,
                    default='visda_src',
                    help='Source dataset name')
parser.add_argument('--tgt',
                    type=str,
                    default='visda_tgt',
                    help='Target dataset name')

# training configurations
parser.add_argument('--batch_size',
                    type=int,
                    default=30,
                    help='Batch size for both training and evaluation')

# logging configurations
parser.add_argument('--log_dir',
                    type=str,
                    default='logs',
                    help='Parent directory for log files')
parser.add_argument('--model_dir',
                    type=str,
                    default=None,
                    help='Model directory for validation')

# resource configurations
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='Selected gpu index')
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of workers')

# model configurations
parser.add_argument('--network',
                    type=str,
                    default='resnet101',  # resnet50
                    help='Base network architecture')


def main():
    args = parser.parse_args()
    print(args)
    config = configure(args.config)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(colored(f"Model directory: {args.model_dir}", 'green'))

    assert os.path.isfile(args.model_dir)

    dataset_name = get_dataset_name(args.src, args.tgt)
    dataset_config = config.data.dataset[dataset_name]
    tgt_file = os.path.join(args.dataset_root, dataset_name, args.tgt + '_list.txt')

    # tgt classification
    model = Model(base_net=args.network,
                  num_classes=dataset_config.num_classes,
                  frozen_layer='')
    del model.classifier_layer
    del model.contrast_layer

    model_state_dict = model.state_dict()
    trained_state_dict = torch.load(args.model_dir)['weights']

    keys = set(model_state_dict.keys())
    trained_keys = set(trained_state_dict.keys())

    shared_keys = keys.intersection(trained_keys)
    to_load_state_dict = {key: trained_state_dict[key] for key in shared_keys}
    model.load_state_dict(to_load_state_dict)
    model = model.cuda()

    # define data loader
    test_data_loader_kwargs = {
        'shuffle': False,
        'drop_last': False,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }

    test_transformer = get_transform(training=False)

    data_loader = {}
    data_iterator = {}
    tgt_test_dataset = DefaultDataset(tgt_file, transform=test_transformer)
    data_loader['tgt_test'] = DataLoader(tgt_test_dataset, **test_data_loader_kwargs)

    for key in data_loader:
        data_iterator[key] = iter(data_loader[key])

    # test
    model.base_network.eval()
    with torch.no_grad():
        tgt_all_features = []
        tgt_all_labels = []
        model.set_bn_domain(domain=1)
        for tgt_test_data in tqdm.tqdm(data_loader['tgt_test'], desc='tgt_test', leave=False, ascii=True):
            tgt_test_inputs, tgt_test_labels = tgt_test_data['image_1'].cuda(), tgt_test_data['true_label'].cuda()
            tgt_test_features = model.base_network(tgt_test_inputs)
            # tgt_test_features = F.normalize(tgt_test_features, p=2, dim=1)
            tgt_all_features += [tgt_test_features]
            tgt_all_labels += [tgt_test_labels]

        tgt_all_features = torch.cat(tgt_all_features, dim=0)
        tgt_all_labels = torch.cat(tgt_all_labels, dim=0)

    tgt_all_features = tgt_all_features.cpu().numpy()
    tgt_all_labels = tgt_all_labels.cpu().numpy()

    features_pickle_file = os.path.join('features', args.src + '_' + args.tgt, 'tgt_features.pkl')
    labels_pickle_file = os.path.join('features', args.src + '_' + args.tgt, 'tgt_labels.pkl')
    pickle.dump(tgt_all_features, open(features_pickle_file, 'wb'))
    pickle.dump(tgt_all_labels, open(labels_pickle_file, 'wb'))


def get_sample(data_loader, data_iterator, data_name):
    try:
        sample = next(data_iterator[data_name])
    except StopIteration:
        data_iterator[data_name] = iter(data_loader[data_name])
        sample = next(data_iterator[data_name])
    return sample


if __name__ == '__main__':
    main()
