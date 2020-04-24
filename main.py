import argparse
import os
import shutil

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from lr_schedule import InvScheduler
from model.costs import LossMultiNCE
from model.key_memory import KeyMemory
from model.model import Model
from train import Train
from utils import configure, get_dataset_name, moment_update

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='config/config.yml',
                    help='Dataset configuration parameters')
parser.add_argument('--src',
                    type=str,
                    default='visda_src',
                    help='Source dataset name')
parser.add_argument('--tgt',
                    type=str,
                    default='visda_tgt',
                    help='Target dataset name')
parser.add_argument('--batch_size',
                    type=int,
                    default=36,
                    help='Batch size for both training and evaluation')
parser.add_argument('--eval_batch_size',
                    type=int,
                    default=36,
                    help='Batch size for both training and evaluation')
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='Selected gpu index')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of workers')
parser.add_argument('--tclip',
                    type=float,
                    default=10.,
                    help='Soft clipping range for NCE scores')
parser.add_argument('--tw',
                    type=float,
                    default=0.0,
                    help='Weight for target classification loss')
parser.add_argument('--cw',
                    type=float,
                    default=1.0,
                    help='Weight for NCE contrast loss')
parser.add_argument('--thresh',
                    type=float,
                    default=0.9,
                    help='Confidence threshold for pseudo labeling target samples')
parser.add_argument('--alpha',
                    type=float,
                    default=0.999,
                    help='momentum coefficient for model ema')
parser.add_argument('--lr_decay',
                    type=float,
                    default=10.,
                    help='learning rate decay coefficient')
parser.add_argument('--min_conf_classes',
                    type=int,
                    default=8,
                    help='minimum number of confident classes')
parser.add_argument('--max_key_size',
                    type=int,
                    default=16384,
                    help='maximum number of key feature size computed in the model')


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
        f"tw_{args.tw}",
        f"cw_{args.cw}",
        f"thresh_{args.thresh}",
        f"min_conf_classes_{args.min_conf_classes}",
        f"max_key_size_{args.max_key_size}",
        f"lr_decay_{args.lr_decay}",
        f"alpha_{args.alpha}",
        f"gpu_{args.gpu}"
    ]
    model_name = "_".join(setup_list)
    print(colored(f"Model name: {model_name}", 'green'))
    model_dir = os.path.join(config.log.log_dir, model_name)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    summary_writer = SummaryWriter(model_dir)

    dataset_name = get_dataset_name(args.src, args.tgt)
    dataset_config = config.data.dataset[dataset_name]
    src_file = os.path.join(config.data.dataset_root, dataset_name, args.src + '_list.txt')
    tgt_file = os.path.join(config.data.dataset_root, dataset_name, args.tgt + '_list.txt')

    model = Model(base_net=dataset_config.network,
                  num_classes=dataset_config.num_classes,
                  bottleneck_dim=dataset_config.bottleneck_dim)
    model_ema = Model(base_net=dataset_config.network,
                      num_classes=dataset_config.num_classes,
                      bottleneck_dim=dataset_config.bottleneck_dim)

    moment_update(model, model_ema, 0)

    model = model.cuda()
    model_ema = model_ema.cuda()

    contrast_loss = LossMultiNCE(tclip=args.tclip).cuda()
    src_memory = KeyMemory(len(open(src_file).readlines()), dataset_config.bottleneck_dim).cuda()
    tgt_memory = KeyMemory(len(open(tgt_file).readlines()), dataset_config.bottleneck_dim).cuda()

    parameters = model.get_parameter_list()
    group_ratios = [parameter['lr'] for parameter in parameters]

    optimizer = torch.optim.SGD(parameters, **config.optimizer.optim.params)
    lr_scheduler = InvScheduler(gamma=config.optimizer.lr_scheduler.gamma,
                                decay_rate=config.optimizer.lr_scheduler.decay_rate,
                                init_lr=config.optimizer.init_lr)

    trainer = Train(model, model_ema, optimizer, lr_scheduler, group_ratios,
                    summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory,
                    tw=args.tw,
                    cw=args.cw,
                    thresh=args.thresh,
                    min_conf_classes=args.min_conf_classes,
                    max_key_size=args.max_key_size,
                    num_classes=dataset_config.num_classes,
                    lr_decay=args.lr_decay,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    max_iter=config.train.max_iteration,
                    iters_per_epoch=dataset_config.iterations_per_epoch,
                    log_summary_interval=config.log.log_summary_interval,
                    log_image_interval=config.log.log_image_interval,
                    eval_interval=config.log.eval_interval,
                    num_proj_samples=config.log.num_proj_samples,
                    alpha=args.alpha)

    tgt_best_acc = trainer.train()
    with open('results.txt', 'a') as f:
        f.write(model_name + '     ' + str(tgt_best_acc) + '\n')
        f.close()


if __name__ == '__main__':
    main()
