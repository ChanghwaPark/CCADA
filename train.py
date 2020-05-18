import os

import torch
import tqdm
from torch import nn

from preprocess.data_provider import DefaultDataLoader, UniformDataLoader, ConfidentDataLoader, NonConfidentDataLoader
from utils import summary_write_proj, summary_write_fig, AvgMeter, moment_update, set_bn_train, compute_accuracy


class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, model_dir,
                 summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory, tgt_pseudo_labeler,
                 cw=1.0,
                 thresh=0.9,
                 num_classes=31,
                 batch_size=36,
                 eval_batch_size=36,
                 num_workers=4,
                 max_iter=100000,
                 iters_per_epoch=100,
                 log_summary_interval=10,
                 log_image_interval=1000,
                 num_proj_samples=384,
                 acc_metric='total_mean',
                 alpha=0.99):
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_dir = model_dir
        self.summary_writer = summary_writer
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.contrast_loss = contrast_loss
        self.src_memory = src_memory
        self.tgt_memory = tgt_memory
        self.tgt_pseudo_labeler = tgt_pseudo_labeler
        self.cw = cw
        self.thresh = thresh
        self.min_conf_samples = 3
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.iters_per_epoch = iters_per_epoch
        self.log_summary_interval = log_summary_interval
        self.log_image_interval = log_image_interval
        self.num_proj_samples = num_proj_samples
        self.acc_metric = acc_metric
        self.alpha = alpha

        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(desc='Iterations', total=self.max_iter, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.acc_dict = {'tgt_best_test_acc': 0.0}
        self.src_train_acc_queue = AvgMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.src_size = len(open(src_file).readlines())
        self.tgt_size = len(open(tgt_file).readlines())
        self.tgt_conf_pair = None
        self.tgt_non_conf_indices = list(range(self.tgt_size))
        self.data_loader = {}
        self.data_iterator = {}
        self.train_data_loader_kwargs = {
            'shuffle': True,
            'drop_last': True,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers
        }
        self.test_data_loader_kwargs = {
            'shuffle': False,
            'drop_last': False,
            'batch_size': self.eval_batch_size,
            'num_workers': self.num_workers
        }
        self.construct_data_loaders()

    def construct_data_loaders(self):
        self.data_loader['src_train'] = UniformDataLoader(self.src_file, self.train_data_loader_kwargs, training=True)
        self.data_loader['tgt_conf'] = None
        self.data_loader['tgt_non_conf'] = None

        self.data_loader['src_embed'] = DefaultDataLoader(self.src_file, self.train_data_loader_kwargs, training=False)
        self.data_loader['tgt_embed'] = DefaultDataLoader(self.tgt_file, self.train_data_loader_kwargs, training=False)

        self.data_loader['src_test'] = DefaultDataLoader(self.src_file, self.test_data_loader_kwargs, training=False)
        self.data_loader['tgt_test'] = DefaultDataLoader(self.tgt_file, self.test_data_loader_kwargs, training=False)

        for key in self.data_loader:
            if self.data_loader[key] is not None:
                self.data_iterator[key] = iter(self.data_loader[key])
            else:
                self.data_iterator[key] = None

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            # update target confident dataset
            self.prepare_tgt_conf_dataset()

            # train an epoch
            self.train_epoch()

        self.total_progress_bar.write('Finish training')
        return self.acc_dict['tgt_best_test_acc']

    def train_epoch(self):
        cur_epoch_steps = max(self.iters_per_epoch, len(self.data_loader['tgt_conf']))
        for _ in tqdm.tqdm(range(cur_epoch_steps), desc='Epoch {:4d}'.format(self.epoch), leave=False, ascii=True):
            self.model.train()
            self.model_ema.eval()
            self.model_ema.apply(set_bn_train)

            self.train_step()

            if self.iteration % self.log_summary_interval == 0:
                self.log_summary_writer()
            if self.iteration % self.log_image_interval == 0:
                self.log_image_writer()

        self.epoch += 1

    def train_step(self):
        # update the learning rate of the optimizer
        # self.optimizer = \
        #     self.lr_scheduler.next_optimizer(self.group_ratios, self.optimizer, self.iteration)
        self.lr_scheduler.adjust_learning_rate(self.optimizer, self.iteration)
        self.optimizer.zero_grad()

        # prepare source batch
        src_data = self.get_sample('src_train')
        # src_inputs, src_labels = src_data['image'].cuda(), src_data['true_label'].cuda()
        src_inputs_1, src_inputs_2, src_labels \
            = src_data['image_1'].cuda(), src_data['image_2'].cuda(), src_data['true_label'].cuda()

        # model inference
        self.model.set_bn_domain(domain=0)
        src_end_points = self.model(src_inputs_1)

        # update key memory
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            src_end_points_ema = self.model_ema(src_inputs_2)
            self.src_memory.store_keys(src_end_points_ema['contrast_features'], src_labels)

        # source classification
        self.src_supervised_step(src_end_points, src_labels)

        if self.data_loader['tgt_conf'].data_loader is not None:
            tgt_data = self.get_sample('tgt_conf')
            # tgt_inputs, tgt_pseudo_labels = tgt_data['image'].cuda(), tgt_data['pseudo_label'].cuda()
            tgt_inputs_1, tgt_inputs_2, tgt_pseudo_labels \
                = tgt_data['image_1'].cuda(), tgt_data['image_2'].cuda(), tgt_data['pseudo_label'].cuda()

            # model inference
            self.model.set_bn_domain(domain=1)
            tgt_end_points = self.model(tgt_inputs_1)

            # update key memory
            with torch.no_grad():
                self.model_ema.set_bn_domain(domain=1)
                tgt_end_points_ema = self.model_ema(tgt_inputs_2)
                self.tgt_memory.store_keys(tgt_end_points_ema['contrast_features'], tgt_pseudo_labels)

            # class contrastive alignment
            self.contrastive_step(src_end_points, src_labels, tgt_end_points, tgt_pseudo_labels)

        else:
            self.losses_dict['contrast_loss'] = 0.

        # pass non conf data for batch norm layers
        if self.data_loader['tgt_non_conf'].data_loader is not None:
            tgt_data = self.get_sample('tgt_non_conf')
            tgt_inputs_1, tgt_inputs_2 = tgt_data['image_1'].cuda(), tgt_data['image_2'].cuda()

            # model inference
            self.model.set_bn_domain(domain=1)
            self.model(tgt_inputs_1)

            with torch.no_grad():
                self.model_ema.set_bn_domain(domain=1)
                self.model_ema(tgt_inputs_2)

        self.losses_dict['total_loss'] = \
            self.losses_dict['src_classification_loss'] \
            + self.cw * self.losses_dict['contrast_loss']
        self.losses_dict['total_loss'].backward()
        self.optimizer.step()

        moment_update(self.model, self.model_ema, self.alpha)

        self.iteration += 1
        self.total_progress_bar.update(1)

    def src_supervised_step(self, src_end_points, src_labels):
        # compute source classification loss
        src_classification_loss = self.class_criterion(src_end_points['logits'], src_labels)
        self.losses_dict['src_classification_loss'] = src_classification_loss

        # compute source train accuracy
        src_train_accuracy = compute_accuracy(src_end_points['logits'], src_labels, acc_metric=self.acc_metric)
        self.src_train_acc_queue.put(src_train_accuracy)

    def contrastive_step(self, src_end_points, src_labels, tgt_end_points=None, tgt_pseudo_labels=None):
        if tgt_end_points is not None:
            batch_features = torch.cat([src_end_points['contrast_features'], tgt_end_points['contrast_features']],
                                       dim=0)
            batch_labels = torch.cat([src_labels, tgt_pseudo_labels], dim=0)
        else:
            batch_features = src_end_points['contrast_features']
            batch_labels = src_labels
        assert batch_labels.lt(0).sum().cpu().numpy() == 0

        src_key_features, src_key_labels = self.src_memory.get_queue()
        tgt_key_features, tgt_key_labels = self.tgt_memory.get_queue()

        key_features = torch.cat([src_key_features, tgt_key_features], dim=0)
        key_labels = torch.cat([src_key_labels, tgt_key_labels], dim=0)

        # (batch_size, key_size)
        pos_matrix = (key_labels == batch_labels.unsqueeze(1)).float()

        # (batch_size, key_size)
        neg_matrix = (key_labels != batch_labels.unsqueeze(1)).float()

        contrast_loss = self.contrast_loss(batch_features, key_features, pos_matrix, neg_matrix)
        self.losses_dict['contrast_loss'] = contrast_loss

    # def non_conf_contrastive_step(self, tgt_end_points, tgt_end_points_ema):
    #     batch_features = tgt_end_points['contrast_features']
    #     batch_features_ema = tgt_end_points_ema['contrast_features']
    #     batch_size = batch_features.size(0)
    #
    #     src_key_features, _ = self.src_memory.get_queue()
    #     tgt_key_features, _ = self.tgt_memory.get_queue()
    #
    #     key_features = torch.cat([src_key_features, tgt_key_features], dim=0)
    #     key_size = key_features.size(0)
    #
    #     key_features = torch.cat([batch_features_ema, key_features], dim=0)
    #
    #     # (batch_size, batch_size + key_size)
    #     pos_matrix_batch = torch.eye(batch_size).float().cuda()
    #     pos_matrix_key = torch.zeros(batch_size, key_size).float().cuda()
    #     pos_matrix = torch.cat([pos_matrix_batch, pos_matrix_key], dim=1)
    #
    #     # (batch_size, batch_size + key_size)
    #     neg_matrix = 1.0 - pos_matrix
    #
    #     non_conf_contrast_loss = self.contrast_loss(batch_features, key_features, pos_matrix, neg_matrix)
    #     self.losses_dict['non_conf_contrast_loss'] = non_conf_contrast_loss

    def prepare_tgt_conf_dataset(self):
        src_test_collection = self.collect_samples('src_test')
        tgt_test_collection = self.collect_samples('tgt_test')
        tgt_pseudo_probabilities = self.tgt_pseudo_labeler.pseudo_label_tgt(src_test_collection, tgt_test_collection)
        tgt_pseudo_acc = compute_accuracy(tgt_pseudo_probabilities, tgt_test_collection['true_labels'],
                                          acc_metric=self.acc_metric, print_result=False)
        self.acc_dict['tgt_pseudo_acc'] = tgt_pseudo_acc
        self.eval_tgt(tgt_test_collection)

        tgt_pseudo_confidences, tgt_pseudo_labels = torch.max(tgt_pseudo_probabilities, dim=1)
        tgt_conf_mask = tgt_pseudo_confidences.ge(self.thresh)
        tgt_conf_indices = torch.tensor(range(self.tgt_size)).cuda()[tgt_conf_mask].tolist()
        tgt_conf_predictions = tgt_pseudo_labels[tgt_conf_mask].tolist()
        self.tgt_conf_pair = list(zip(tgt_conf_indices, tgt_conf_predictions))

        self.data_loader['tgt_conf'] = ConfidentDataLoader(
            self.tgt_file, self.train_data_loader_kwargs, self.tgt_conf_pair, self.min_conf_samples, training=True)

        if self.data_loader['tgt_conf'].data_loader is None:
            self.data_iterator['tgt_conf'] = None
        else:
            self.data_iterator['tgt_conf'] = iter(self.data_loader['tgt_conf'])

        tgt_non_conf_mask = tgt_pseudo_confidences.lt(self.thresh)
        self.tgt_non_conf_indices = torch.tensor(range(self.tgt_size)).cuda()[tgt_non_conf_mask].tolist()

        self.data_loader['tgt_non_conf'] = NonConfidentDataLoader(
            self.tgt_file, self.train_data_loader_kwargs, self.tgt_non_conf_indices, training=True)

        if self.data_loader['tgt_non_conf'].data_loader is None:
            self.data_iterator['tgt_non_conf'] = None
        else:
            self.data_iterator['tgt_non_conf'] = iter(self.data_loader['tgt_non_conf'])

    def eval_tgt(self, tgt_test_collection):
        tgt_test_acc = compute_accuracy(tgt_test_collection['logits'], tgt_test_collection['true_labels'],
                                        acc_metric=self.acc_metric, print_result=False)
        tgt_test_acc = round(tgt_test_acc, 3)
        self.acc_dict['tgt_test_acc'] = tgt_test_acc
        self.acc_dict['tgt_best_test_acc'] = max(self.acc_dict['tgt_best_test_acc'], tgt_test_acc)
        self.print_acc()
        self.save_checkpoint()

    def collect_samples(self, data_name):
        assert 'src' in data_name or 'tgt' in data_name
        if 'src' in data_name:
            domain = 0
        else:
            domain = 1

        self.model_ema.eval()
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=domain)
            sample_collection = {}
            sample_features = []
            sample_logits = []
            sample_true_labels = []

            for sample_data in tqdm.tqdm(self.data_loader[data_name], desc=data_name, leave=False, ascii=True):
                batch_inputs, batch_true_labels = sample_data['image_1'].cuda(), sample_data['true_label'].cuda()
                batch_end_points = self.model_ema(batch_inputs)

                sample_features += [batch_end_points['features']]
                sample_logits += [batch_end_points['logits']]
                sample_true_labels += [batch_true_labels]

            sample_collection['features'] = torch.cat(sample_features, dim=0)
            sample_collection['logits'] = torch.cat(sample_logits, dim=0)
            sample_collection['true_labels'] = torch.cat(sample_true_labels, dim=0)

        return sample_collection

    def get_sample(self, data_name):
        try:
            sample = next(self.data_iterator[data_name])
        except StopIteration:
            if data_name == 'src_train' or data_name == 'tgt_conf':
                self.data_loader[data_name].construct_data_loader()
            self.data_iterator[data_name] = iter(self.data_loader[data_name])
            sample = next(self.data_iterator[data_name])
        except TypeError:
            assert self.data_loader[data_name].data_loader is None
            return None
        return sample

    def log_summary_writer(self):
        self.summary_writer.add_scalars('losses', self.losses_dict, global_step=self.iteration)
        self.summary_writer.add_scalars('accuracies', self.acc_dict, global_step=self.iteration)
        self.summary_writer.close()

    def log_image_writer(self):
        src_data = next(iter(self.data_loader['src_embed']))
        src_inputs, src_labels = src_data['image_1'].cuda(), src_data['true_label'].cuda()
        tgt_data = next(iter(self.data_loader['tgt_embed']))
        tgt_inputs, tgt_labels = tgt_data['image_1'].cuda(), tgt_data['true_label'].cuda()
        summary_write_fig(self.summary_writer, tag='Source predictions vs. true labels',
                          global_step=self.iteration,
                          model=self.model, images=src_inputs, labels=src_labels, domain=0)
        summary_write_fig(self.summary_writer, tag='Target predictions vs. true labels',
                          global_step=self.iteration,
                          model=self.model, images=tgt_inputs, labels=tgt_labels, domain=1)
        summary_write_proj(self.summary_writer, tag='features', global_step=self.iteration,
                           model=self.model,
                           src_train_loader=self.data_loader['src_embed'],
                           tgt_train_loader=self.data_loader['tgt_embed'],
                           num_samples=self.num_proj_samples)
        summary_write_proj(self.summary_writer, tag='logits', global_step=self.iteration,
                           model=self.model,
                           src_train_loader=self.data_loader['src_embed'],
                           tgt_train_loader=self.data_loader['tgt_embed'],
                           num_samples=self.num_proj_samples)

    def print_acc(self):
        # show the latest eval_result
        self.acc_dict['src_train_accuracy'] = self.src_train_acc_queue.get_average()
        self.total_progress_bar.write('Iteration {:6d}: '.format(self.iteration) + str(self.acc_dict))

    def save_checkpoint(self):
        checkpoint_weights = os.path.join(self.model_dir, 'checkpoint_%d_%d.weights' % (self.epoch, self.iteration))
        torch.save({'weights': self.model.state_dict()}, checkpoint_weights)
