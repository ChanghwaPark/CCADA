import argparse
import os

import torch
import torch.nn as nn
import tqdm
from termcolor import colored
from torch.utils.data import DataLoader

from lr_schedule import InvScheduler
from model.model import Model
from model.utils import initialize_layer
from preprocess.data_provider import get_transform
from preprocess.default_dataset import DefaultDataset
from utils import configure, get_dataset_name, str2bool, compute_accuracy

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
parser.add_argument('--eval_batch_size',
                    type=int,
                    default=30,
                    help='Batch size for both training and evaluation')
parser.add_argument('--max_iterations',
                    type=int,
                    default=10000,
                    help='Maximum number of iterations')
parser.add_argument('--print_acc_interval',
                    type=int,
                    default=100,
                    help='Print accuracy interval while training')

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

# optimizer configurations
parser.add_argument('--optimizer',
                    type=str,
                    default='sgd',
                    help='Optimizer type')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='Initial learning rate')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='Optimizer parameter, momentum')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0005,
                    help='Optimizer parameter, weight decay')
parser.add_argument('--nesterov',
                    type=str2bool,
                    default=False,  # True
                    help='Optimizer parameter, nesterov')

# learning rate scheduler configurations
parser.add_argument('--lr_scheduler',
                    type=str,
                    default='inv',
                    help='Learning rate scheduler type')
parser.add_argument('--gamma',
                    type=float,
                    default=0.001,  # 0.0005
                    help='Inv learning rate scheduler parameter, gamma')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.75,  # 2.25
                    help='Inv learning rate scheduler parameter, decay rate')


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
    src_file = os.path.join(args.dataset_root, dataset_name, args.src + '_list.txt')
    tgt_file = os.path.join(args.dataset_root, dataset_name, args.tgt + '_list.txt')

    # src classification
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

    # source classifier
    src_classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.base_network.out_dim, dataset_config.num_classes)
    )
    initialize_layer(src_classifier)
    parameter_list = [{"params": src_classifier.parameters(), "lr": 1}]
    src_classifier = src_classifier.cuda()

    group_ratios = [parameter['lr'] for parameter in parameter_list]
    optimizer = torch.optim.SGD(parameter_list,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    assert args.lr_scheduler == 'inv'
    lr_scheduler = InvScheduler(gamma=args.gamma,
                                decay_rate=args.decay_rate,
                                group_ratios=group_ratios,
                                init_lr=args.lr)

    # define data loaders
    train_data_loader_kwargs = {
        'shuffle': True,
        'drop_last': True,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    test_data_loader_kwargs = {
        'shuffle': False,
        'drop_last': False,
        'batch_size': args.eval_batch_size,
        'num_workers': args.num_workers
    }

    train_transformer = get_transform(training=True)
    test_transformer = get_transform(training=False)

    data_loader = {}
    data_iterator = {}
    src_train_dataset = DefaultDataset(src_file, transform=train_transformer)
    data_loader['src_train'] = DataLoader(src_train_dataset, **train_data_loader_kwargs)
    src_test_dataset = DefaultDataset(src_file, transform=test_transformer)
    data_loader['src_test'] = DataLoader(src_test_dataset, **test_data_loader_kwargs)

    for key in data_loader:
        data_iterator[key] = iter(data_loader[key])

    # start training
    total_progress_bar = tqdm.tqdm(desc='Iterations', total=args.max_iterations, ascii=True, smoothing=0.01)
    class_criterion = nn.CrossEntropyLoss()
    model.base_network.eval()
    src_classifier.train()
    iteration = 0
    while iteration < args.max_iterations:
        lr_scheduler.adjust_learning_rate(optimizer, iteration)
        optimizer.zero_grad()

        src_data = get_sample(data_loader, data_iterator, 'src_train')
        src_inputs, src_labels = src_data['image_1'].cuda(), src_data['true_label'].cuda()

        model.set_bn_domain(domain=0)
        with torch.no_grad():
            src_features = model.base_network(src_inputs)
            # src_features = F.normalize(src_features, p=2, dim=1)
        src_class_logits = src_classifier(src_features)

        src_classification_loss = class_criterion(src_class_logits, src_labels)

        if iteration % args.print_acc_interval == 0:
            compute_accuracy(src_class_logits, src_labels, acc_metric=dataset_config.acc_metric, print_result=True)

        total_loss = src_classification_loss
        total_loss.backward()
        optimizer.step()

        iteration += 1
        total_progress_bar.update(1)

    # test
    model.base_network.eval()
    src_classifier.eval()
    with torch.no_grad():
        src_all_class_logits = []
        src_all_labels = []
        model.set_bn_domain(domain=0)
        for src_test_data in tqdm.tqdm(data_loader['src_test'], desc='src_test', leave=False, ascii=True):
            src_test_inputs, src_test_labels = src_test_data['image_1'].cuda(), src_test_data['true_label'].cuda()
            src_test_features = model.base_network(src_test_inputs)
            # src_test_features = F.normalize(src_test_features, p=2, dim=1)
            src_test_class_logits = src_classifier(src_test_features)

            src_all_class_logits += [src_test_class_logits]
            src_all_labels += [src_test_labels]

        src_all_class_logits = torch.cat(src_all_class_logits, dim=0)
        src_all_labels = torch.cat(src_all_labels, dim=0)

        compute_accuracy(src_all_class_logits, src_all_labels, acc_metric='total_mean', print_result=True)
        compute_accuracy(src_all_class_logits, src_all_labels, acc_metric='class_mean', print_result=True)

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

    # source classifier
    tgt_classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.base_network.out_dim, dataset_config.num_classes)
    )
    initialize_layer(tgt_classifier)
    parameter_list = [{"params": tgt_classifier.parameters(), "lr": 1}]
    tgt_classifier = tgt_classifier.cuda()

    group_ratios = [parameter['lr'] for parameter in parameter_list]
    optimizer = torch.optim.SGD(parameter_list,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    assert args.lr_scheduler == 'inv'
    lr_scheduler = InvScheduler(gamma=args.gamma,
                                decay_rate=args.decay_rate,
                                group_ratios=group_ratios,
                                init_lr=args.lr)

    # define data loaders
    train_data_loader_kwargs = {
        'shuffle': True,
        'drop_last': True,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    test_data_loader_kwargs = {
        'shuffle': False,
        'drop_last': False,
        'batch_size': args.eval_batch_size,
        'num_workers': args.num_workers
    }

    train_transformer = get_transform(training=True)
    test_transformer = get_transform(training=False)

    data_loader = {}
    data_iterator = {}
    tgt_train_dataset = DefaultDataset(tgt_file, transform=train_transformer)
    data_loader['tgt_train'] = DataLoader(tgt_train_dataset, **train_data_loader_kwargs)
    tgt_test_dataset = DefaultDataset(tgt_file, transform=test_transformer)
    data_loader['tgt_test'] = DataLoader(tgt_test_dataset, **test_data_loader_kwargs)

    for key in data_loader:
        data_iterator[key] = iter(data_loader[key])

    # start training
    total_progress_bar = tqdm.tqdm(desc='Iterations', total=args.max_iterations, ascii=True, smoothing=0.01)
    class_criterion = nn.CrossEntropyLoss()
    model.base_network.eval()
    tgt_classifier.train()
    iteration = 0
    while iteration < args.max_iterations:
        lr_scheduler.adjust_learning_rate(optimizer, iteration)
        optimizer.zero_grad()

        tgt_data = get_sample(data_loader, data_iterator, 'tgt_train')
        tgt_inputs, tgt_labels = tgt_data['image_1'].cuda(), tgt_data['true_label'].cuda()

        model.set_bn_domain(domain=1)
        with torch.no_grad():
            tgt_features = model.base_network(tgt_inputs)
            # tgt_features = F.normalize(tgt_features, p=2, dim=1)
        tgt_class_logits = tgt_classifier(tgt_features)

        tgt_classification_loss = class_criterion(tgt_class_logits, tgt_labels)

        if iteration % args.print_acc_interval == 0:
            compute_accuracy(tgt_class_logits, tgt_labels, acc_metric=dataset_config.acc_metric, print_result=True)

        total_loss = tgt_classification_loss
        total_loss.backward()
        optimizer.step()

        iteration += 1
        total_progress_bar.update(1)

    # test
    model.base_network.eval()
    tgt_classifier.eval()
    with torch.no_grad():
        tgt_all_class_logits = []
        tgt_all_labels = []
        model.set_bn_domain(domain=1)
        for tgt_test_data in tqdm.tqdm(data_loader['tgt_test'], desc='tgt_test', leave=False, ascii=True):
            tgt_test_inputs, tgt_test_labels = tgt_test_data['image_1'].cuda(), tgt_test_data['true_label'].cuda()
            tgt_test_features = model.base_network(tgt_test_inputs)
            # tgt_test_features = F.normalize(tgt_test_features, p=2, dim=1)
            tgt_test_class_logits = tgt_classifier(tgt_test_features)

            tgt_all_class_logits += [tgt_test_class_logits]
            tgt_all_labels += [tgt_test_labels]

        tgt_all_class_logits = torch.cat(tgt_all_class_logits, dim=0)
        tgt_all_labels = torch.cat(tgt_all_labels, dim=0)

        compute_accuracy(tgt_all_class_logits, tgt_all_labels, acc_metric='total_mean', print_result=True)
        compute_accuracy(tgt_all_class_logits, tgt_all_labels, acc_metric='class_mean', print_result=True)


def get_sample(data_loader, data_iterator, data_name):
    try:
        sample = next(data_iterator[data_name])
    except StopIteration:
        data_iterator[data_name] = iter(data_loader[data_name])
        sample = next(data_iterator[data_name])
    return sample


if __name__ == '__main__':
    main()
