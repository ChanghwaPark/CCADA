import argparse
import json
import os
import shutil

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from lr_schedule import InvScheduler
from model.contrastive_loss import InfoNCELoss
from model.key_memory import KeyMemory
from model.model import Model
from train import Train
from utils import configure, get_dataset_name, moment_update, str2bool

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
                    default=36,
                    help='Batch size for both training and evaluation')
parser.add_argument('--eval_batch_size',
                    type=int,
                    default=36,
                    help='Batch size for both training and evaluation')
parser.add_argument('--pseudo_batch_size',
                    type=int,
                    default=4096,
                    help='Batch size for pseudo labeling')
parser.add_argument('--max_iterations',
                    type=int,
                    default=10000,
                    help='Maximum number of iterations')

# logging configurations
parser.add_argument('--log_dir',
                    type=str,
                    default='logs',
                    help='Parent directory for log files')
parser.add_argument('--log_summary_interval',
                    type=int,
                    default=10,
                    help='Logging summaries frequency')
parser.add_argument('--log_image_interval',
                    type=int,
                    default=1000,
                    help='Logging images frequency')
parser.add_argument('--num_project_samples',
                    type=int,
                    default=384,
                    help='Number of samples for tensorboard projection')
parser.add_argument('--acc_file',
                    type=str,
                    default='result.txt',
                    help='File where accuracies are wrote')

# resource configurations
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='Selected gpu index')
parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Number of workers')

# InfoNCE loss configurations
parser.add_argument('--tclip',
                    type=float,
                    default=20.,
                    help='Soft clipping range for NCE scores')
parser.add_argument('--contrast_normalize',
                    type=str2bool,
                    default=True,
                    help='Feature normalization for the contrastive loss computation')

# hyper-parameters
parser.add_argument('--cw',
                    type=float,
                    default=1.0,
                    help='Weight for NCE contrast loss')
parser.add_argument('--thresh',
                    type=float,
                    default=0.9,
                    help='Confidence threshold for pseudo labeling target samples')
parser.add_argument('--min_conf_classes',
                    type=int,
                    default=8,
                    help='Minimum number of confident classes')
parser.add_argument('--max_key_size',
                    type=int,
                    default=16384,
                    help='Maximum number of key feature size computed in the model')
parser.add_argument('--pseudo_labeling',
                    type=str,
                    default='kmeans',
                    help='Pseudo labeling method. Choose between ["info", "kmeans", "lp", "classifier"]')
parser.add_argument('--pseudo_normalize',
                    type=str2bool,
                    default=False,
                    help='Feature normalization for the pseudo labeling. "info" and "lp" use this.')

# model configurations
parser.add_argument('--network',
                    type=str,
                    default='resnet50',
                    help='Base network architecture')
parser.add_argument('--bottleneck',
                    type=str2bool,
                    default=False,
                    help='Use bottleneck layer between the base network and the classifier layer.')
parser.add_argument('--bottleneck_dim',
                    type=int,
                    default=2048,
                    help='bottleneck layer dimension')
parser.add_argument('--alpha',
                    type=float,
                    default=0.9,
                    help='momentum coefficient for model ema')
parser.add_argument('--frozen_layer',
                    type=str,
                    default='layer1',
                    help='Frozen layer in the base network')

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

    # define model name
    setup_list = [
        args.src,
        args.tgt,
        args.network,
        f"bottleneck_{args.bottleneck}_dim_{args.bottleneck_dim}",
        f"alpha_{args.alpha}",
        f"tclip_{args.tclip}_normalize_{args.contrast_normalize}",
        f"cw_{args.cw}",
        f"thresh_{args.thresh}",
        f"min_conf_classes_{args.min_conf_classes}",
        f"pseudo_labeling_{args.pseudo_labeling}_normalize_{args.pseudo_normalize}",
        f"gpu_{args.gpu}"
    ]
    model_name = "_".join(setup_list)
    print(colored(f"Model name: {model_name}", 'green'))
    model_dir = os.path.join(args.log_dir, model_name)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    summary_writer = SummaryWriter(model_dir)

    # save parsed arguments
    with open(os.path.join(model_dir, 'parsed_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    dataset_name = get_dataset_name(args.src, args.tgt)
    dataset_config = config.data.dataset[dataset_name]
    src_file = os.path.join(args.dataset_root, dataset_name, args.src + '_list.txt')
    tgt_file = os.path.join(args.dataset_root, dataset_name, args.tgt + '_list.txt')

    model = Model(base_net=args.network,
                  num_classes=dataset_config.num_classes,
                  bottleneck=args.bottleneck,
                  bottleneck_dim=args.bottleneck_dim,
                  frozen_layer=args.frozen_layer)
    model_ema = Model(base_net=args.network,
                      num_classes=dataset_config.num_classes,
                      bottleneck=args.bottleneck,
                      bottleneck_dim=args.bottleneck_dim,
                      frozen_layer=args.frozen_layer)

    moment_update(model, model_ema, 0)

    model = model.cuda()
    model_ema = model_ema.cuda()

    contrast_loss = InfoNCELoss(tclip=args.tclip, feat_normalize=args.contrast_normalize).cuda()
    src_memory = KeyMemory(len(open(src_file).readlines()), args.bottleneck_dim).cuda()
    tgt_memory = KeyMemory(len(open(tgt_file).readlines()), args.bottleneck_dim).cuda()

    if args.pseudo_labeling == 'info':
        from pseudo_labeler import InfoPseudoLabeler
        tgt_pseudo_labeler = InfoPseudoLabeler(num_classes=dataset_config.num_classes,
                                               batch_size=args.pseudo_batch_size,
                                               tclip=args.tclip,
                                               normalize=args.pseudo_normalize)
    elif args.pseudo_labeling == 'kmeans':
        from pseudo_labeler import KMeansPseudoLabeler
        tgt_pseudo_labeler = KMeansPseudoLabeler(num_classes=dataset_config.num_classes,
                                                 batch_size=args.pseudo_batch_size)
    elif args.pseudo_labeling == 'lp':
        from pseudo_labeler import PropagatePseudoLabeler
        tgt_pseudo_labeler = PropagatePseudoLabeler(num_classes=dataset_config.num_classes,
                                                    batch_size=args.pseudo_batch_size,
                                                    tclip=args.tclip,
                                                    normalize=args.pseudo_normalize)
    elif args.pseudo_labeling == 'classifier':
        from pseudo_labeler import ClassifierPseudoLabeler
        tgt_pseudo_labeler = ClassifierPseudoLabeler(num_classes=dataset_config.num_classes)
    else:
        raise ValueError

    parameters = model.get_parameter_list()
    group_ratios = [parameter['lr'] for parameter in parameters]

    optimizer = torch.optim.SGD(parameters,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    assert args.lr_scheduler == 'inv'
    lr_scheduler = InvScheduler(gamma=args.gamma,
                                decay_rate=args.decay_rate,
                                group_ratios=group_ratios,
                                init_lr=args.lr)

    trainer = Train(model, model_ema, optimizer, lr_scheduler,
                    summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory, tgt_pseudo_labeler,
                    cw=args.cw,
                    thresh=args.thresh,
                    min_conf_classes=args.min_conf_classes,
                    max_key_size=args.max_key_size,
                    num_classes=dataset_config.num_classes,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    max_iter=args.max_iterations,
                    iters_per_epoch=dataset_config.iterations_per_epoch,
                    log_summary_interval=args.log_summary_interval,
                    log_image_interval=args.log_image_interval,
                    num_proj_samples=args.num_project_samples,
                    acc_metric=dataset_config.acc_metric,
                    alpha=args.alpha)

    tgt_best_acc = trainer.train()
    with open(args.acc_file, 'a') as f:
        f.write(model_name + '     ' + str(tgt_best_acc) + '\n')
        f.close()


if __name__ == '__main__':
    main()
