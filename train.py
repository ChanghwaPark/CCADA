import torch
import tqdm
from torch import nn

from preprocess.data_provider import get_data_loader, get_certain_data_loader, get_uniform_data_loader
from utils import summary_write_embeddings, summary_write_figures, AverageMeter, moment_update, get_labels_from_file


class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, group_ratios,
                 summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory,
                 tgt_weight=1.0,
                 contrast_weight=1.0,
                 threshold=0.7,
                 confident_classes=12,
                 num_classes=31,
                 lr_decay=5,
                 batch_size=36,
                 eval_batch_size=36,
                 num_workers=4,
                 is_center=False,
                 max_iter=100000,
                 iterations_per_epoch=10,
                 log_scalar_interval=1,
                 print_interval=10,
                 log_image_interval=100,
                 num_embedding_samples=384,
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
        self.src_memory = src_memory
        self.tgt_memory = tgt_memory
        self.tgt_weight = tgt_weight
        self.contrast_weight = contrast_weight
        self.threshold = threshold
        self.confident_classes = confident_classes
        self.num_classes = num_classes
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.is_center = is_center
        self.max_iter = max_iter
        self.iterations_per_epoch = iterations_per_epoch
        self.log_scalar_interval = log_scalar_interval
        self.print_interval = print_interval
        self.log_image_interval = log_image_interval
        self.num_embedding_samples = num_embedding_samples
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
                            'tgt_certain': None,
                            'src_test': get_data_loader(self.src_file, self.test_data_loader_kwargs, training=False),
                            'tgt_test': get_data_loader(self.tgt_file, self.test_data_loader_kwargs, training=False),
                            'src_embed': get_data_loader(self.src_file, self.train_data_loader_kwargs, training=False),
                            'tgt_embed': get_data_loader(self.tgt_file, self.train_data_loader_kwargs, training=False)}
        self.data_iterator = {'src_train': iter(self.data_loader['src_train']),
                              'tgt_train': iter(self.data_loader['tgt_train']),
                              'tgt_certain': None,
                              'src_test': iter(self.data_loader['src_test']),
                              'tgt_test': iter(self.data_loader['tgt_test'])}
        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(
            desc='Iterations', total=self.max_iter, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.accuracies_dict = {}
        self.src_train_accuracy_queue = AverageMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.src_size = len(open(src_file).readlines())
        self.tgt_size = len(open(tgt_file).readlines())
        self.src_all_labels = torch.tensor(get_labels_from_file(src_file)).cuda()
        self.tgt_all_pseudo_labels = torch.tensor([-1] * self.tgt_size).cuda()
        self.tgt_best_test_accuracy = 0.
        self.min_dataset_size = self.batch_size * self.iterations_per_epoch

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            # evaluation and update pseudo labels update after each epoch
            self.tgt_test()

            # update tgt_certain dataset
            self.prepare_tgt_certain_dataset()

            # train an epoch
            self.train_epoch()

            # log to summary_writer
            self.log_summary_writer()

        self.total_progress_bar.write('Finish training')

    def train_epoch(self):
        for _ in tqdm.tqdm(range(self.iterations_per_epoch),
                           desc='Epoch {:4d}'.format(self.epoch), leave=False, ascii=True):
            self.model.train()
            self.model_ema.eval()

            def set_bn_train(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.train()

            self.model_ema.apply(set_bn_train)

            self.train_step()
        self.epoch += 1

    def train_step(self):

        # update the learning rate of the optimizer
        self.optimizer = \
            self.lr_scheduler.next_optimizer(self.group_ratios, self.optimizer, self.iteration / self.lr_decay)
        self.optimizer.zero_grad()

        # prepare source and target batches
        src_inputs, src_labels, src_indices = self.get_sample('src_train')

        if self.data_loader['tgt_certain'] is not None:
            tgt_inputs, tgt_indices = self.get_sample('tgt_certain')
        else:
            tgt_inputs, _, tgt_indices = self.get_sample('tgt_train')

        src_inputs, src_labels, src_indices, tgt_inputs, tgt_indices \
            = src_inputs.cuda(), src_labels.cuda(), src_indices.cuda(), tgt_inputs.cuda(), tgt_indices.cuda()

        # tgt pseudo labels
        tgt_pseudo_labels = torch.gather(self.tgt_all_pseudo_labels, dim=0, index=tgt_indices)

        # model inference
        self.model.set_bn_domain(domain=0)
        src_end_points = self.model(src_inputs)
        self.model.set_bn_domain(domain=1)
        tgt_end_points = self.model(tgt_inputs)

        # update key memory
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            src_end_points_ema = self.model_ema(src_inputs)
            self.src_memory.store_keys(src_end_points_ema['features'], src_indices)

            self.model_ema.set_bn_domain(domain=1)
            tgt_end_points_ema = self.model_ema(tgt_inputs)
            self.tgt_memory.store_keys(tgt_end_points_ema['features'], tgt_indices)

        # source classification
        self.src_supervised_step(src_end_points, src_labels)

        if self.data_loader['tgt_certain'] is not None:
            # target pseudo classification
            self.tgt_supervised_step(tgt_end_points, tgt_pseudo_labels)

            # class contrastive alignment
            self.contrastive_step(src_end_points, src_labels, tgt_end_points, tgt_pseudo_labels)
        else:
            self.losses_dict['tgt_classification_loss'] = 0.
            self.losses_dict['query_to_key_loss'] = 0.
            self.losses_dict['contrast_norm_loss'] = 0.
            self.losses_dict['contrast_loss'] = 0.

        self.losses_dict['total_loss'] = \
            self.losses_dict['src_classification_loss'] \
            + self.tgt_weight * self.losses_dict['tgt_classification_loss'] \
            + self.contrast_weight * self.losses_dict['contrast_loss']
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
        self.src_train_accuracy_queue.put(src_train_accuracy)

    def tgt_supervised_step(self, tgt_end_points, tgt_pseudo_labels):
        tgt_classification_loss = self.class_criterion(tgt_end_points['logits'], tgt_pseudo_labels)
        self.losses_dict['tgt_classification_loss'] = tgt_classification_loss

    def contrastive_step(self, src_end_points, src_labels, tgt_end_points, tgt_pseudo_labels):
        batch_features = torch.cat([src_end_points['features'], tgt_end_points['features']], dim=0)

        src_features_key = self.src_memory.get_queue()
        tgt_features_key = self.tgt_memory.get_queue()
        src_features_key, tgt_features_key = src_features_key.cuda(), tgt_features_key.cuda()
        all_features_key = torch.cat([src_features_key, tgt_features_key], dim=0)

        all_labels = torch.cat([self.src_all_labels, self.tgt_all_pseudo_labels], dim=0)
        batch_labels = torch.cat([src_labels, tgt_pseudo_labels], dim=0)
        active_batch_mask = batch_labels.ge(0)
        active_all_mask = all_labels.ge(0)

        # (batch_size, key_size)
        active_mask_matrix = active_all_mask & active_batch_mask.unsqueeze(1)

        # (batch_size, key_size)
        pos_matrix = (all_labels == batch_labels.unsqueeze(1))
        pos_matrix = (pos_matrix & active_mask_matrix).float()

        # (batch_size, key_size)
        neg_matrix = (all_labels != batch_labels.unsqueeze(1))
        neg_matrix = (neg_matrix & active_mask_matrix).float()

        query_to_key_loss, contrast_norm_loss = \
            self.contrast_loss(batch_features, all_features_key, pos_matrix, neg_matrix)
        self.losses_dict['query_to_key_loss'] = query_to_key_loss
        self.losses_dict['contrast_norm_loss'] = contrast_norm_loss

        contrast_loss = query_to_key_loss + contrast_norm_loss
        self.losses_dict['contrast_loss'] = contrast_loss

    def tgt_test(self):
        self.model_ema.eval()
        with torch.no_grad():
            tgt_all_predictions = torch.zeros(self.tgt_size).long().cuda()
            tgt_all_confidences = torch.zeros(self.tgt_size).cuda()

            self.model_ema.set_bn_domain(domain=1)
            correct = 0
            for (tgt_inputs, tgt_labels, tgt_indices) in tqdm.tqdm(
                    self.data_loader['tgt_test'], desc='Evaluation', leave=False, ascii=True):
                tgt_inputs, tgt_labels, tgt_indices = tgt_inputs.cuda(), tgt_labels.cuda(), tgt_indices.cuda()
                tgt_end_points = self.model_ema(tgt_inputs)
                # self.tgt_memory.store_keys(tgt_end_points['features'], tgt_indices)  # TODO

                correct += (tgt_end_points['predictions'] == tgt_labels).sum().item()

                tgt_all_predictions.index_copy_(0, tgt_indices, tgt_end_points['predictions'])
                tgt_all_confidences.index_copy_(0, tgt_indices, tgt_end_points['confidences'])
            self.accuracies_dict['tgt_test_accuracy'] = round(correct / len(self.data_loader['tgt_test'].dataset), 5)

            if self.tgt_best_test_accuracy < self.accuracies_dict['tgt_test_accuracy']:
                self.tgt_best_test_accuracy = self.accuracies_dict['tgt_test_accuracy']

            # update tgt_all_pseudo_labels
            tgt_all_confident_mask = tgt_all_confidences.ge(self.threshold)
            tgt_all_uncertain_labels = torch.tensor([-1] * self.tgt_size).cuda()
            self.tgt_all_pseudo_labels = torch.where(
                tgt_all_confident_mask, tgt_all_predictions, tgt_all_uncertain_labels)

    def prepare_tgt_certain_dataset(self):
        tgt_all_pseudo_labels_list = self.tgt_all_pseudo_labels.cpu().tolist()
        confident_classes = list(set(tgt_all_pseudo_labels_list))
        if -1 in confident_classes:
            confident_classes.remove(-1)
        # if self.tgt_all_pseudo_labels.ge(0).sum() < self.batch_size:  # TODO
        if len(confident_classes) < self.confident_classes:
            self.data_loader['tgt_certain'] = None
            self.data_iterator['tgt_certain'] = None
            return

        self.data_loader['tgt_certain'] = get_certain_data_loader(
            self.tgt_file, self.train_data_loader_kwargs, self.tgt_all_pseudo_labels,
            min_dataset_size=self.min_dataset_size, is_center=self.is_center)
        self.data_iterator['tgt_certain'] = iter(self.data_loader['tgt_certain'])

    def get_sample(self, data_name):
        try:
            sample = next(self.data_iterator[data_name])
        except StopIteration:
            if data_name == 'src_train':
                self.data_loader[data_name] = get_uniform_data_loader(self.src_file, self.train_data_loader_kwargs,
                                                                      is_center=self.is_center)
            self.data_iterator[data_name] = iter(self.data_loader[data_name])
            sample = next(self.data_iterator[data_name])
        except TypeError:
            assert self.data_loader[data_name] is None
            return None
        return sample

    def log_summary_writer(self):
        if self.epoch % self.log_scalar_interval == 0:
            self.summary_writer.add_scalars('losses', self.losses_dict, global_step=self.iteration)
            self.summary_writer.add_scalars('accuracies', self.accuracies_dict, global_step=self.iteration)
            self.summary_writer.close()

        if self.epoch % self.log_image_interval == 0:
            src_inputs, src_labels, _ = next(iter(self.data_loader['src_train']))
            tgt_inputs, tgt_labels, _ = next(iter(self.data_loader['tgt_train']))
            src_inputs, src_labels, tgt_inputs, tgt_labels = \
                src_inputs.cuda(), src_labels.cuda(), tgt_inputs.cuda(), tgt_labels.cuda()
            summary_write_figures(self.summary_writer, tag='Source predictions vs. true labels',
                                  global_step=self.iteration,
                                  model=self.model, images=src_inputs, labels=src_labels, domain=0)
            summary_write_figures(self.summary_writer, tag='Target predictions vs. true labels',
                                  global_step=self.iteration,
                                  model=self.model, images=tgt_inputs, labels=tgt_labels, domain=1)
            summary_write_embeddings(self.summary_writer, tag='features', global_step=self.iteration,
                                     model=self.model,
                                     src_train_loader=self.data_loader['src_embed'],
                                     tgt_train_loader=self.data_loader['tgt_embed'],
                                     num_samples=self.num_embedding_samples)
            summary_write_embeddings(self.summary_writer, tag='logits', global_step=self.iteration,
                                     model=self.model,
                                     src_train_loader=self.data_loader['src_embed'],
                                     tgt_train_loader=self.data_loader['tgt_embed'],
                                     num_samples=self.num_embedding_samples)
            self.model.train()

        if self.epoch % self.print_interval == 0:
            # show the latest eval_result
            self.accuracies_dict['src_train_accuracy'] = self.src_train_accuracy_queue.get_average()
            self.total_progress_bar.write('Iteration {:6d}: '.format(self.iteration) + str(self.accuracies_dict))
            self.total_progress_bar.write(f'Target best test accuracy: {self.tgt_best_test_accuracy}')
