import torch
import tqdm
from torch import nn

from preprocess.data_provider import get_data_loader, get_conf_data_loader, get_uniform_data_loader
from utils import summary_write_proj, summary_write_fig, AvgMeter, moment_update, set_bn_train


class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, group_ratios,
                 summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory,
                 tw=0.0,
                 cw=1.0,
                 thresh=0.9,
                 min_conf_classes=10,
                 max_key_size=16384,
                 num_classes=31,
                 lr_decay=5,
                 batch_size=36,
                 eval_batch_size=36,
                 num_workers=4,
                 is_center=False,
                 max_iter=100000,
                 iters_per_epoch=100,
                 log_summary_interval=10,
                 log_image_interval=1000,
                 eval_interval=1000,
                 num_proj_samples=384,
                 alpha=0.99):
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.group_ratios = group_ratios
        self.summary_writer = summary_writer
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.contrast_loss = contrast_loss
        self.max_key_size = max_key_size
        self.src_memory = src_memory
        self.tgt_memory = tgt_memory
        self.tw = tw
        self.cw = cw
        self.thresh = thresh
        self.min_conf_classes = min_conf_classes
        self.num_classes = num_classes
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.is_center = is_center
        self.max_iter = max_iter
        self.iters_per_epoch = iters_per_epoch
        self.log_summary_interval = log_summary_interval
        self.log_image_interval = log_image_interval
        self.eval_interval = eval_interval
        self.num_proj_samples = num_proj_samples
        self.alpha = alpha

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
        self.data_loader = {'src_train': get_uniform_data_loader(self.src_file, self.train_data_loader_kwargs,
                                                                 is_center=self.is_center),
                            'tgt_train': get_data_loader(self.tgt_file, self.train_data_loader_kwargs,
                                                         training=True, is_center=self.is_center),
                            'tgt_conf': None,
                            # 'src_test': get_data_loader(self.src_file, self.test_data_loader_kwargs, training=False),
                            'tgt_test': get_data_loader(self.tgt_file, self.test_data_loader_kwargs, training=False),
                            'src_embed': get_data_loader(self.src_file, self.train_data_loader_kwargs, training=False),
                            'tgt_embed': get_data_loader(self.tgt_file, self.train_data_loader_kwargs, training=False)}
        self.data_iterator = {'src_train': iter(self.data_loader['src_train']),
                              'tgt_train': iter(self.data_loader['tgt_train']),
                              'tgt_conf': None,
                              # 'src_test': iter(self.data_loader['src_test']),
                              'tgt_test': iter(self.data_loader['tgt_test'])}
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

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            # train an epoch
            self.train_epoch()

            # update target confident dataset
            self.prepare_tgt_conf_dataset()

        self.total_progress_bar.write('Finish training')
        return self.acc_dict['tgt_best_test_acc']

    def train_epoch(self):
        # iterations_per_epoch = self.get_iter_per_epoch()

        for _ in tqdm.tqdm(range(self.iters_per_epoch), desc='Epoch {:4d}'.format(self.epoch), leave=False, ascii=True):
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
        self.optimizer = \
            self.lr_scheduler.next_optimizer(self.group_ratios, self.optimizer, self.iteration / self.lr_decay)
        self.optimizer.zero_grad()

        # prepare source batch
        src_inputs, src_labels = self.get_sample('src_train')
        src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()

        # model inference
        self.model.set_bn_domain(domain=0)
        src_end_points = self.model(src_inputs)

        # update key memory
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            src_end_points_ema = self.model_ema(src_inputs)
            self.src_memory.store_keys(src_end_points_ema['features'], src_labels)

        # source classification
        self.src_supervised_step(src_end_points, src_labels)

        if self.data_loader['tgt_conf'] is not None:
            tgt_inputs, tgt_pseudo_labels = self.get_sample('tgt_conf')
            tgt_inputs, tgt_pseudo_labels = tgt_inputs.cuda(), tgt_pseudo_labels.cuda()

            # model inference
            self.model.set_bn_domain(domain=1)
            tgt_end_points = self.model(tgt_inputs)

            # update key memory
            with torch.no_grad():
                self.model_ema.set_bn_domain(domain=1)
                tgt_end_points_ema = self.model_ema(tgt_inputs)
                self.tgt_memory.store_keys(tgt_end_points_ema['features'], tgt_pseudo_labels)

            if self.tw > 0:
                # target pseudo classification
                self.tgt_supervised_step(tgt_end_points, tgt_pseudo_labels)
            else:
                self.losses_dict['tgt_classification_loss'] = 0.

            # class contrastive alignment
            self.contrastive_step(src_end_points, src_labels, tgt_end_points, tgt_pseudo_labels)

        else:
            self.losses_dict['tgt_classification_loss'] = 0.
            self.losses_dict['query_to_key_loss'] = 0.
            self.losses_dict['contrast_norm_loss'] = 0.
            self.losses_dict['contrast_loss'] = 0.
            # self.contrastive_step(src_end_points, src_labels)

        self.losses_dict['total_loss'] = \
            self.losses_dict['src_classification_loss'] \
            + self.tw * self.losses_dict['tgt_classification_loss'] \
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
        src_train_accuracy = (src_end_points['predictions'] == src_labels).sum().item() / src_labels.size(0)
        self.src_train_acc_queue.put(src_train_accuracy)

    def tgt_supervised_step(self, tgt_end_points, tgt_pseudo_labels):
        tgt_classification_loss = self.class_criterion(tgt_end_points['logits'], tgt_pseudo_labels)
        self.losses_dict['tgt_classification_loss'] = tgt_classification_loss

    def contrastive_step(self, src_end_points, src_labels, tgt_end_points=None, tgt_pseudo_labels=None):
        if tgt_end_points is not None:
            batch_features = torch.cat([src_end_points['features'], tgt_end_points['features']], dim=0)
            batch_labels = torch.cat([src_labels, tgt_pseudo_labels], dim=0)
        else:
            batch_features = src_end_points['features']
            batch_labels = src_labels
        assert batch_labels.lt(0).sum().cpu().numpy() == 0

        # src_updated_indices = torch.tensor(range(self.src_size)).cuda()[self.src_updated_flag]
        # if self.max_key_feature_size < len(src_updated_indices):
        #     src_selected_indices = torch.tensor(
        #         random.sample(src_updated_indices.tolist(), self.max_key_feature_size)).cuda()
        # else:
        #     src_selected_indices = src_updated_indices
        src_key_features, src_key_labels = self.src_memory.get_queue()

        # if self.max_key_feature_size < len(self.tgt_certain_indices):
        #     tgt_selected_indices = torch.tensor(
        #         random.sample(self.tgt_certain_indices.tolist(), self.max_key_feature_size)).cuda()
        # else:
        #     tgt_selected_indices = self.tgt_certain_indices
        tgt_key_features, tgt_key_labels = self.tgt_memory.get_queue()

        key_features = torch.cat([src_key_features, tgt_key_features], dim=0)

        # src_key_labels = torch.index_select(self.src_all_labels, dim=0, index=src_selected_indices)
        # tgt_key_labels = torch.index_select(self.tgt_all_pseudo_labels, dim=0, index=tgt_selected_indices)

        key_labels = torch.cat([src_key_labels, tgt_key_labels], dim=0)

        # (batch_size, key_size)
        pos_matrix = (key_labels == batch_labels.unsqueeze(1)).float()

        # (batch_size, key_size)
        neg_matrix = (key_labels != batch_labels.unsqueeze(1)).float()

        query_to_key_loss, contrast_norm_loss = \
            self.contrast_loss(batch_features, key_features, pos_matrix, neg_matrix)
        self.losses_dict['query_to_key_loss'] = query_to_key_loss
        self.losses_dict['contrast_norm_loss'] = contrast_norm_loss

        contrast_loss = query_to_key_loss + contrast_norm_loss
        self.losses_dict['contrast_loss'] = contrast_loss

    # def tgt_test(self):
    # self.model_ema.eval()
    # with torch.no_grad():
    # tgt_all_predictions = torch.zeros(self.tgt_size).long().cuda()
    # tgt_all_confidences = torch.zeros(self.tgt_size).cuda()

    # self.model_ema.set_bn_domain(domain=1)
    # correct = 0
    # for (tgt_inputs, tgt_labels, _) in tqdm.tqdm(
    #         self.data_loader['tgt_test'], desc='Evaluation', leave=False, ascii=True):
    #     tgt_inputs, tgt_labels = tgt_inputs.cuda(), tgt_labels.cuda()
    #     tgt_end_points = self.model_ema(tgt_inputs)
    # self.tgt_memory.store_keys(tgt_end_points['features'], tgt_indices)  # TODO

    # correct += (tgt_end_points['predictions'] == tgt_labels).sum().item()

    # tgt_all_predictions.index_copy_(0, tgt_indices, tgt_end_points['predictions'])
    # tgt_all_confidences.index_copy_(0, tgt_indices, tgt_end_points['confidences'])
    # tgt_test_acc = round(correct / len(self.data_loader['tgt_test'].dataset), 5)
    # tgt_test_acc, *_ = self.evaluate_tgt()
    # self.acc_dict['tgt_test_acc'] = tgt_test_acc
    # self.acc_dict['tgt_best_test_acc'] = max(self.acc_dict['tgt_best_test_acc'], tgt_test_acc)

    # if self.accuracies_dict['tgt_best_test_acc'] < self.accuracies_dict['tgt_test_acc']:
    #     self.accuracies_dict['tgt_best_test_acc'] = self.accuracies_dict['tgt_test_acc']

    # update tgt_all_pseudo_labels
    # tgt_all_confident_mask = tgt_all_confidences.ge(self.thresh)
    # tgt_all_uncertain_labels = torch.tensor([-1] * self.tgt_size).cuda()
    # self.tgt_all_pseudo_labels = torch.where(
    #     tgt_all_confident_mask, tgt_all_predictions, tgt_all_uncertain_labels)
    # tgt_certain_indices_list = [index for (index, pseudo_label) in enumerate(self.tgt_all_pseudo_labels)
    #                             if pseudo_label >= 0]
    # self.tgt_certain_indices = torch.tensor(tgt_certain_indices_list).cuda()

    def prepare_tgt_conf_dataset(self):
        # tgt_conf_pair = self.eval_tgt()
        self.eval_tgt()
        self.data_loader['tgt_conf'] = get_conf_data_loader(
            self.tgt_file, self.train_data_loader_kwargs, self.tgt_conf_pair,
            self.min_conf_classes, is_center=self.is_center)

        if self.data_loader['tgt_conf'] is None:
            self.data_iterator['tgt_conf'] = None
        else:
            self.data_iterator['tgt_conf'] = iter(self.data_loader['tgt_conf'])

    def eval_tgt(self):
        self.model_ema.eval()
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=1)
            correct = 0
            tgt_conf_indices = []
            tgt_conf_predictions = []

            for (tgt_inputs, tgt_labels, tgt_indices) in tqdm.tqdm(
                    self.data_loader['tgt_test'], desc='Target labeling', leave=False, ascii=True):
                tgt_inputs, tgt_labels, tgt_indices = tgt_inputs.cuda(), tgt_labels.cuda(), tgt_indices.cuda()
                tgt_end_points = self.model_ema(tgt_inputs)

                correct += (tgt_end_points['predictions'] == tgt_labels).sum().item()

                tgt_conf_mask = tgt_end_points['confidences'].ge(self.thresh)
                tgt_conf_indices.extend(tgt_indices[tgt_conf_mask].tolist())
                tgt_conf_predictions.extend(tgt_end_points['predictions'][tgt_conf_mask].tolist())

        print('tgt_conf_indices, tgt_conf_predictions')
        assert len(tgt_conf_indices) == len(tgt_conf_predictions)
        print(f'len(tgt_conf_indices): {len(tgt_conf_indices)}')
        # print(list(zip(tgt_conf_indices, tgt_conf_predictions)))

        tgt_test_acc = round(correct / len(self.data_loader['tgt_test'].dataset), 5)
        self.acc_dict['tgt_test_acc'] = tgt_test_acc
        self.acc_dict['tgt_best_test_acc'] = max(self.acc_dict['tgt_best_test_acc'], tgt_test_acc)

        self.print_acc()

        # tgt_conf_pair = list(zip(tgt_conf_indices, tgt_conf_predictions))
        self.tgt_conf_pair = list(zip(tgt_conf_indices, tgt_conf_predictions))
        # return tgt_conf_pair

        # if self.accuracies_dict['tgt_best_test_acc'] < self.accuracies_dict['tgt_test_acc']:
        #     self.accuracies_dict['tgt_best_test_acc'] = self.accuracies_dict['tgt_test_acc']

        # update tgt_all_pseudo_labels
        # tgt_all_confident_mask = tgt_all_confidences.ge(self.thresh)
        # tgt_all_uncertain_labels = torch.tensor([-1] * self.tgt_size).cuda()
        # self.tgt_all_pseudo_labels = torch.where(
        #     tgt_all_confident_mask, tgt_all_predictions, tgt_all_uncertain_labels)
        # tgt_certain_indices_list = [index for (index, pseudo_label) in enumerate(self.tgt_all_pseudo_labels)
        #                             if pseudo_label >= 0]
        # self.tgt_certain_indices = torch.tensor(tgt_certain_indices_list).cuda()

    # def get_iter_per_epoch(self):
    #     if self.data_loader['tgt_conf'] is None:
    #         print(f'self.default_iter_per_epoch: {self.iters_per_epoch}')
    #         return self.iters_per_epoch
    #     else:
    #         print(f'len(self.data_loader["tgt_conf"]): {len(self.data_loader["tgt_conf"])}')
    #         return len(self.data_loader['tgt_conf'])

    def get_sample(self, data_name):
        try:
            sample = next(self.data_iterator[data_name])
        except StopIteration:
            if data_name == 'src_train':
                self.data_loader[data_name] = \
                    get_uniform_data_loader(self.src_file, self.train_data_loader_kwargs, is_center=self.is_center)
            # if data_name == 'tgt_conf':
            #     self.data_loader[data_name] = \
            #         get_certain_data_loader(self.tgt_file, self.train_data_loader_kwargs, self.tgt_all_pseudo_labels,
            #                                 num_confident_samples=self.num_confident_samples, is_center=self.is_center)
            if data_name == 'tgt_conf':
                self.data_loader[data_name] = get_conf_data_loader(
                    self.tgt_file, self.train_data_loader_kwargs, self.tgt_conf_pair,
                    self.min_conf_classes, is_center=self.is_center)
            self.data_iterator[data_name] = iter(self.data_loader[data_name])
            sample = next(self.data_iterator[data_name])
        except TypeError:
            assert self.data_loader[data_name] is None
            return None
        return sample

    def log_summary_writer(self):
        self.summary_writer.add_scalars('losses', self.losses_dict, global_step=self.iteration)
        self.summary_writer.add_scalars('accuracies', self.acc_dict, global_step=self.iteration)
        self.summary_writer.close()

        # if self.iteration % self.log_image_interval == 0:

    def log_image_writer(self):
        src_inputs, src_labels = next(iter(self.data_loader['src_train']))
        tgt_inputs, tgt_labels, _ = next(iter(self.data_loader['tgt_train']))
        src_inputs, src_labels, tgt_inputs, tgt_labels = \
            src_inputs.cuda(), src_labels.cuda(), tgt_inputs.cuda(), tgt_labels.cuda()
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
        # self.model.train()

        # if self.iteration % self.eval_interval == 0:

    def print_acc(self):
        # show the latest eval_result
        self.acc_dict['src_train_accuracy'] = self.src_train_acc_queue.get_average()
        self.total_progress_bar.write('Iteration {:6d}: '.format(self.iteration) + str(self.acc_dict))
        # self.total_progress_bar.write(f'Target best test accuracy: {self.tgt_best_test_accuracy}')
