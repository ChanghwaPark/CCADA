import torch
import tqdm
from torch import nn

from preprocess.data_provider import get_data_loader, get_confident_data_loader
from utils import summary_write_embeddings, summary_write_figures, prepare_categorical_data, AverageMeter, moment_update


class Train:
    def __init__(self, model, model_ema, optimizer, lr_scheduler, group_ratios,
                 summary_writer, src_file, tgt_file, contrast_loss, key_memory,
                 dim_weight=1.0,
                 threshold=0.5,
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
        self.key_memory = key_memory
        self.dim_weight = dim_weight
        self.threshold = threshold
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
                            'tgt_test': get_data_loader(self.tgt_file, self.test_data_loader_kwargs, training=False),
                            'src_embed': get_data_loader(self.src_file, self.train_data_loader_kwargs, training=False),
                            'tgt_embed': get_data_loader(self.tgt_file, self.train_data_loader_kwargs, training=False)}
        self.data_iterator = {'src_train': iter(self.data_loader['src_train']),
                              'tgt_train': iter(self.data_loader['tgt_train']),
                              'tgt_test': iter(self.data_loader['tgt_test'])}
        self.iteration = 0
        self.epoch = 0
        self.total_progress_bar = tqdm.tqdm(
            desc='Iterations', total=self.max_iter, ncols=120, ascii=True, smoothing=0.01)
        self.losses_dict = {}
        self.accuracies_dict = {}
        self.src_train_accuracy_queue = AverageMeter(maxsize=100)
        self.class_criterion = nn.CrossEntropyLoss()
        self.tgt_predictions_list = [-1] * len(self.data_loader['tgt_test'].dataset)
        self.tgt_confidences_list = [0.] * len(self.data_loader['tgt_test'].dataset)

    def train(self):
        # start training
        self.total_progress_bar.write('Start training')

        while self.iteration < self.max_iter:
            self.train_epoch()

            # evaluation after each epoch
            self.tgt_predictions_list, self.tgt_confidences_list = self.tgt_test()

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
                summary_write_embeddings(self.summary_writer, tag='feature', global_step=self.iteration,
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
        self.optimizer = self.lr_scheduler.next_optimizer(self.group_ratios, self.optimizer, self.iteration / 5)
        # TODO
        self.optimizer.zero_grad()

        # source classification
        src_inputs, src_labels, _ = self.get_sample('src_train')
        tgt_inputs, _, tgt_indices = self.get_sample('tgt_train')
        src_inputs, src_labels, tgt_inputs = src_inputs.cuda(), src_labels.cuda(), tgt_inputs.cuda()
        self.src_supervised_step(src_inputs, src_labels)

        # local deep infomax
        self.contrastive_step(src_inputs, src_labels, tgt_inputs, tgt_indices)

        self.losses_dict['total_loss'] = \
            self.losses_dict['src_classification_loss'] + self.dim_weight * self.losses_dict['dim_loss']
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

    def local_deep_infomax_step(self, src_inputs, src_labels, tgt_inputs, tgt_labels, memory_features, memory_labels):
        labels = torch.cat((src_labels, tgt_labels), dim=0)
        memory_features, memory_labels = memory_features.cuda(), memory_labels.cuda()

        self.model.set_bn_domain(domain=0)
        src_end_points_1 = self.model(src_inputs)
        self.model.set_bn_domain(domain=1)
        tgt_end_points_1 = self.model(tgt_inputs)

        r1 = torch.cat([src_end_points_1['r1'], tgt_end_points_1['r1']], dim=0)

        with torch.no_grad():
            self.model_ema.set_bn_domain(domain=0)
            src_end_points_2 = self.model_ema(src_inputs)
            self.model_ema.set_bn_domain(domain=1)
            tgt_end_points_2 = self.model_ema(tgt_inputs)

            local_feature = torch.cat(
                [src_end_points_2[self.local_feature_name], tgt_end_points_2[self.local_feature_name]], dim=0)
            if len(local_feature.size()) == 2:
                local_feature = local_feature.unsqueeze(dim=2).unsqueeze(dim=3)

            all_local_features = torch.cat([local_feature, memory_features], dim=0)
            all_labels = torch.cat([labels, memory_labels], dim=0).unsqueeze(dim=1)

            mask_mat = (all_labels == all_labels.transpose(0, 1)).float().cuda()  # TODO
            mask_mat = mask_mat.narrow(dim=0, start=0, length=labels.size(0))
            # mask_mat = torch.eye(labels.size(0)).cuda()

        loss_1tn, lgt_reg = self.contrast_loss(r1, all_local_features, mask_mat)
        self.losses_dict['loss_1tn'] = loss_1tn
        self.losses_dict['lgt_reg'] = lgt_reg

        dim_loss = loss_1tn + lgt_reg
        self.losses_dict['dim_loss'] = dim_loss

    def tgt_test(self):
        predictions_list = []
        confidences_list = []
        self.model_ema.set_bn_domain(domain=1)
        self.model_ema.eval()
        with torch.no_grad():
            correct = 0
            for (inputs, labels, _) in tqdm.tqdm(self.data_loader['tgt_test'],
                                                 desc='Evaluation', ncols=120, leave=False, ascii=True):
                inputs, labels = inputs.cuda(), labels.cuda()
                end_points = self.model_ema(inputs)

                predictions_list.extend(end_points['predictions'].tolist())
                confidences_list.extend(end_points['confidences'])

                correct += (end_points['predictions'] == labels).sum().item()
            self.accuracies_dict['tgt_test_accuracy'] = round(correct / len(self.data_loader['tgt_test'].dataset), 5)
        return predictions_list, confidences_list

    def update_key_memory(self):
        with torch.no_grad():
            # src_inputs, src_labels = self.get_sample('src_queue')
            src_inputs, src_labels, _ = self.get_sample('src_queue')
            # tgt_inputs, _ = self.get_sample('tgt_queue')
            tgt_inputs, _, tgt_indices = self.get_sample('tgt_queue')
            tgt_labels = [self.tgt_predictions_list[index] if self.tgt_confidences_list[index] > self.threshold
                          else -1 for index in tgt_indices.tolist()]
            # src_size = src_inputs.size(0)
            # tgt_size = tgt_inputs.size(0)
            src_inputs, tgt_inputs, src_labels = src_inputs.cuda(), tgt_inputs.cuda(), src_labels.cuda()
            # tgt_inputs = tgt_inputs.cuda()
            # inputs = torch.cat((src_inputs, tgt_inputs), dim=0)
            self.model_ema.set_bn_domain(domain=0)
            src_end_points = self.model_ema(src_inputs)
            self.model_ema.set_bn_domain(domain=1)
            tgt_end_points = self.model_ema(tgt_inputs)

            # end_points = self.model_ema(inputs)
            # _, tgt_pseudo_labels = torch.split(end_points['predictions'], [src_size, tgt_size], dim=0)
            # tgt_labels = [tgt_pseudo_labels[index] if end_points['confidences'][src_size + index] > self.threshold
            #               else -1 for index in range(tgt_size)]
            # tgt_pseudo_labels = tgt_end_points['predictions'].tolist()
            # tgt_labels = [tgt_pseudo_labels[index] if tgt_end_points['confidences'][index] > self.threshold
            #               else -1 for index in range(tgt_size)]
            tgt_labels = torch.tensor(tgt_labels).cuda()
            labels = torch.cat([src_labels, tgt_labels], dim=0)

            # local_feature = end_points[self.local_feature_name]
            local_feature = torch.cat(
                [src_end_points[self.local_feature_name], tgt_end_points[self.local_feature_name]], dim=0)
            # local_feature = tgt_end_points[self.local_feature_name]
            # if len(end_points[self.local_feature_name].size()) == 2:
            if len(local_feature.size()) == 2:
                local_feature = local_feature.unsqueeze(dim=2).unsqueeze(dim=3)

            # self.key_memory.store_keys(end_points[self.local_feature_name], labels)
            self.key_memory.store_keys(local_feature, labels)
            # self.key_memory.store_keys(local_feature, tgt_labels)

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
