import numpy as np
import torch
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn

from preprocess.data_provider import get_data_loader
from utils import summary_write_embeddings, summary_write_figures, AverageMeter, moment_update


class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, group_ratios,
                 summary_writer, src_file, tgt_file, contrast_loss, src_memory, tgt_memory,
                 contrast_weight=1.0,
                 # knn_samples=3,
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
        self.contrast_weight = contrast_weight
        # self.knn_samples = knn_samples
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
        self.data_loader = {'src_train': get_data_loader(self.src_file, self.train_data_loader_kwargs,
                                                         training=True, is_center=self.is_center),
                            'tgt_train': get_data_loader(self.tgt_file, self.train_data_loader_kwargs,
                                                         training=True, is_center=self.is_center),
                            'src_test': get_data_loader(self.src_file, self.test_data_loader_kwargs, training=False),
                            'tgt_test': get_data_loader(self.tgt_file, self.test_data_loader_kwargs, training=False),
                            'src_embed': get_data_loader(self.src_file, self.train_data_loader_kwargs, training=False),
                            'tgt_embed': get_data_loader(self.tgt_file, self.train_data_loader_kwargs, training=False)}
        self.data_iterator = {'src_train': iter(self.data_loader['src_train']),
                              'tgt_train': iter(self.data_loader['tgt_train']),
                              'src_test': iter(self.data_loader['src_test']),
                              'tgt_test': iter(self.data_loader['tgt_test'])}
        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(
            desc='Iterations', total=self.max_iter, ncols=120, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.accuracies_dict = {}
        self.src_train_accuracy_queue = AverageMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.src_size = len(open(src_file).readlines())
        self.tgt_size = len(open(tgt_file).readlines())
        # self.pos_neg_matrix = torch.zeros((self.src_size + self.tgt_size), (self.src_size + self.tgt_size)).cuda()
        # self.clustered_components = np.zeros((self.src_size + self.tgt_size), dtype=int)
        self.clustered_components = torch.zeros((self.src_size + self.tgt_size)).cuda()

        self.src_sanity = False

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            # evaluation and update clustered components update after each epoch
            self.tgt_test_and_update_matrix()

            # train an epoch
            self.train_epoch()

            # log to summary_writer
            self.log_summary_writer()

        self.total_progress_bar.write('Finish training')

    def train_epoch(self):
        for _ in tqdm.tqdm(range(self.iterations_per_epoch),
                           desc='Epoch {:4d}'.format(self.epoch), ncols=120, leave=False, ascii=True):
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
        tgt_inputs, _, tgt_indices = self.get_sample('tgt_train')
        src_inputs, src_labels, src_indices, tgt_inputs, tgt_indices \
            = src_inputs.cuda(), src_labels.cuda(), src_indices.cuda(), tgt_inputs.cuda(), tgt_indices.cuda()

        # source classification
        self.src_supervised_step(src_inputs, src_labels)

        # class contrastive alignment
        self.contrastive_step(src_inputs, src_indices, tgt_inputs, tgt_indices)

        self.losses_dict['total_loss'] = \
            self.losses_dict['src_classification_loss'] + self.contrast_weight * self.losses_dict['contrast_loss']
        self.losses_dict['total_loss'].backward()
        self.optimizer.step()

        moment_update(self.model, self.model_ema, self.alpha)

        self.iteration += 1
        self.total_progress_bar.update(1)

    def src_supervised_step(self, src_inputs, src_labels):
        self.model.set_bn_domain(domain=0)
        end_points = self.model(src_inputs)
        src_logits = end_points['logits']

        # compute source classification loss
        src_classification_loss = self.class_criterion(src_logits, src_labels)
        self.losses_dict['src_classification_loss'] = src_classification_loss

        # compute source train accuracy
        _, src_predicted = torch.max(src_logits.data, 1)
        src_train_accuracy = (src_predicted == src_labels).sum().item() / src_labels.size(0)
        self.src_train_accuracy_queue.put(src_train_accuracy)

    def contrastive_step(self, src_inputs, src_indices, tgt_inputs, tgt_indices):
        self.model.set_bn_domain(domain=0)
        src_end_points = self.model(src_inputs)
        self.model.set_bn_domain(domain=1)
        tgt_end_points = self.model(tgt_inputs)

        batch_features = torch.cat([src_end_points['features'], tgt_end_points['features']], dim=0)

        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            src_end_points_ema = self.model_ema(src_inputs)
            self.src_memory.store_keys(src_end_points_ema['features'], src_indices)

            self.model_ema.set_bn_domain(domain=1)
            tgt_end_points_ema = self.model_ema(tgt_inputs)
            self.tgt_memory.store_keys(tgt_end_points_ema['features'], tgt_indices)

        src_features_key = self.src_memory.get_queue()
        tgt_features_key = self.tgt_memory.get_queue()
        src_features_key, tgt_features_key = src_features_key.cuda(), tgt_features_key.cuda()
        all_features_key = torch.cat([src_features_key, tgt_features_key], dim=0)

        tgt_indices = tgt_indices + self.src_size
        batch_indices = torch.cat([src_indices, tgt_indices], dim=0)
        batch_components = self.clustered_components.gather(dim=0, index=batch_indices)
        active_batch_components = batch_components.ge(0)
        active_all_components = self.clustered_components.ge(0)

        # (batch_size, key_size)
        active_mask_matrix = active_all_components & active_batch_components.unsqueeze(1)

        # (batch_size, key_size)
        pos_matrix = (self.clustered_components == batch_components.unsqueeze(1))
        pos_matrix = (pos_matrix & active_mask_matrix).float()

        # (batch_size, key_size)
        neg_matrix = (self.clustered_components != batch_components.unsqueeze(1))
        neg_matrix = (neg_matrix & active_mask_matrix).float()

        query_to_key_loss, contrast_norm_loss = \
            self.contrast_loss(batch_features, all_features_key, pos_matrix, neg_matrix)
        self.losses_dict['query_to_key_loss'] = query_to_key_loss
        self.losses_dict['contrast_norm_loss'] = contrast_norm_loss

        contrast_loss = query_to_key_loss + contrast_norm_loss
        self.losses_dict['contrast_loss'] = contrast_loss

    def tgt_test_and_update_matrix(self):
        src_all_features = torch.zeros(self.src_memory.get_size()).cuda()
        src_all_labels = torch.zeros(self.src_memory.get_size()[0]).long().cuda()
        src_all_predictions = torch.zeros(self.src_memory.get_size()[0]).long().cuda()
        src_all_confidences = []
        tgt_all_features = torch.zeros(self.tgt_memory.get_size()).cuda()

        self.model_ema.eval()
        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            for (src_inputs, src_labels, src_indices) in tqdm.tqdm(
                    self.data_loader['src_test'], desc='Source update', ncols=120, leave=False, ascii=True):
                src_inputs, src_labels, src_indices = src_inputs.cuda(), src_labels.cuda(), src_indices.cuda()
                src_end_points = self.model_ema(src_inputs)
                src_all_features.index_copy_(0, src_indices, src_end_points['features'])
                src_all_labels.index_copy_(0, src_indices, src_labels)
                src_all_predictions.index_copy_(0, src_indices, src_end_points['predictions'])
                src_all_confidences.extend(src_end_points['confidences'])

            self.model_ema.set_bn_domain(domain=1)
            correct = 0
            for (tgt_inputs, tgt_labels, tgt_indices) in tqdm.tqdm(
                    self.data_loader['tgt_test'], desc='Evaluation', ncols=120, leave=False, ascii=True):
                tgt_inputs, tgt_labels, tgt_indices = tgt_inputs.cuda(), tgt_labels.cuda(), tgt_indices.cuda()
                tgt_end_points = self.model_ema(tgt_inputs)
                tgt_all_features.index_copy_(0, tgt_indices, tgt_end_points['features'])
                correct += (tgt_end_points['predictions'] == tgt_labels).sum().item()
            self.accuracies_dict['tgt_test_accuracy'] = round(correct / len(self.data_loader['tgt_test'].dataset), 5)

            self.update_connected_components(
                src_all_features, src_all_labels, src_all_predictions, src_all_confidences, tgt_all_features)

    def update_connected_components(self, src_all_features, src_all_labels, src_all_predictions, src_all_confidences,
                                    tgt_all_features):
        """
        update clustered components for contrast loss
        Args:
            src_all_features: aggregated source features
            src_all_labels: aggregated source true labels
            src_all_predictions: aggregated source predictions
            tgt_all_features: aggregated target features
        """
        assert self.src_size == src_all_features.size(0) == src_all_labels.size(0) == src_all_predictions.size(0)
        assert self.tgt_size == tgt_all_features.size(0)
        assert src_all_features.size(1) == tgt_all_features.size(1)

        src_all_confidences = torch.tensor(src_all_confidences).cuda()
        final_component_labels = np.concatenate((src_all_labels.cpu().numpy(), np.array([-1] * self.tgt_size)))

        # src_right_indices = (src_all_predictions == src_all_labels).nonzero(as_tuple=True)[0]
        src_right_indices = \
        ((src_all_predictions == src_all_labels) & src_all_confidences.ge(0.)).nonzero(as_tuple=True)[0]
        src_right_features = src_all_features.index_select(dim=0, index=src_right_indices)
        src_right_labels = src_all_labels.index_select(dim=0, index=src_right_indices).unsqueeze(dim=1)
        src_right_confidences = src_all_confidences.index_select(dim=0, index=src_right_indices)
        print(src_right_labels.size(0))
        if src_right_labels.unique().size(0) < self.num_classes:
            self.clustered_components = torch.tensor(final_component_labels).cuda()
            return

        self.src_sanity_check(src_right_features, src_right_labels)
        # print(self.src_sanity)

        # src_all_labels = src_all_labels.unsqueeze(dim=1)

        # # src_src_connections = (src_all_labels == src_all_labels.transpose(0, 1))
        src_src_connections = (src_right_labels == src_right_labels.transpose(0, 1))
        # # src_tgt_scores = torch.mm(src_all_features, tgt_all_features.transpose(0, 1)).float()
        # src_tgt_scores = torch.mm(src_right_features, tgt_all_features.transpose(0, 1)).float()
        # tgt_src_scores = src_tgt_scores.transpose(0, 1)
        # tgt_tgt_scores = torch.mm(tgt_all_features, tgt_all_features.transpose(0, 1)).float()
        src_right_tgt_features = torch.cat([src_right_features, tgt_all_features], dim=0)
        # all_scores = torch.mm(src_right_tgt_features, src_right_tgt_features.transpose(0, 1)).float()
        all_scores = torch.mm(src_right_features, src_right_features.transpose(0, 1)).float()

        knn_samples = 0
        continue_clustering = True
        while continue_clustering:
            # src_tgt_connections = self.get_connections(src_tgt_scores, knn_samples)
            # tgt_src_connections = self.get_connections(tgt_src_scores, knn_samples)
            # tgt_tgt_connections = self.get_connections(tgt_tgt_scores, knn_samples)

            # all_connections = torch.cat([torch.cat([src_src_connections, src_tgt_connections], dim=1),
            #                              torch.cat([tgt_src_connections, tgt_tgt_connections], dim=1)], dim=0)
            all_connections = self.get_connections(all_scores, knn_samples)
            all_connections = all_connections.cpu().numpy()

            all_graph = csr_matrix(all_connections)
            n_components, component_labels = connected_components(csgraph=all_graph, directed=True, connection='weak')
            print(n_components)
            print(component_labels)

            continue_clustering = self.check_components(n_components, component_labels, src_src_connections,
                                                        src_right_confidences)
            print(continue_clustering)

            if continue_clustering:
                final_component_labels = component_labels
                knn_samples += 1

        self.clustered_components = torch.tensor(final_component_labels).cuda()

    def src_sanity_check(self, src_all_features, src_all_labels):
        """
        check if source features are clustered and aligned with source true labels
        Args:
            src_all_features: aggregated source features
            src_all_labels: aggregated source true labels
        """
        assert src_all_features.size(0) == src_all_labels.size(0)

        raw_scores = torch.mm(src_all_features, src_all_features.transpose(0, 1)).float()
        max_score = torch.max(raw_scores)
        min_score = torch.min(raw_scores)

        src_same_matrix = (src_all_labels == src_all_labels.transpose(1, 0))
        src_diff_matrix = ~src_same_matrix
        src_same_scores = raw_scores * src_same_matrix + max_score * src_diff_matrix
        src_diff_scores = raw_scores * src_diff_matrix + min_score * src_same_matrix
        src_same_min_scores = torch.min(src_same_scores, dim=1)[0]
        src_diff_max_scores = torch.max(src_diff_scores, dim=1)[0]
        self.src_sanity = torch.all(src_same_min_scores.eq(torch.max(src_same_min_scores, src_diff_max_scores))).item()

    def check_components(self, n_components, component_labels, src_src_connections, src_right_confidences):
        if n_components < self.num_classes:
            return False

        # src_component_labels = component_labels[:self.src_size]
        src_component_labels = component_labels[:src_src_connections.size(0)]
        src_component_labels = np.expand_dims(src_component_labels, 1)
        src_component_connections = (src_component_labels == src_component_labels.transpose())
        src_src_connections = src_src_connections.cpu().numpy()

        print(src_src_connections.shape)
        print(src_component_connections.shape)
        false_connections = np.argwhere((src_src_connections >= src_component_connections) == False)
        print(false_connections)
        false_indices = np.unique(false_connections)
        print(false_indices)
        src_right_confidences = src_right_confidences.cpu().numpy()
        false_confidences = src_right_confidences[false_indices]
        print(false_confidences)
        print(src_right_confidences.mean())
        print(false_confidences.mean())
        print(false_connections.shape)
        print(false_connections)
        # return np.array_equal(src_component_connections, src_src_connections)
        return np.all(src_src_connections >= src_component_connections)

    @staticmethod
    def get_connections(scores, k):
        assert scores.size(0) == scores.size(1)
        min_scores = torch.min(scores, dim=1, keepdim=True)[0]
        masked_scores = torch.eye(scores.size(0)).cuda() * min_scores + (1. - torch.eye(scores.size(0)).cuda()) * scores
        # indices = torch.topk(scores, k=k, dim=1)[1]
        if k > 0:
            indices = torch.topk(masked_scores, k=k, dim=1)[1]
            connections = torch.zeros_like(scores).bool().scatter_(dim=1, index=indices, src=torch.tensor(True)).cuda()
            return connections
        else:
            return torch.eye(scores.size(0)).bool().cuda()

    def get_sample(self, data_name):
        try:
            sample = next(self.data_iterator[data_name])
        except StopIteration:
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
