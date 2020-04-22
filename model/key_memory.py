import math

import torch
from torch import nn


class KeyMemory(nn.Module):
    def __init__(self, queue_size, feature_dim):
        super(KeyMemory, self).__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.index = 0

        stdv = 1. / math.sqrt(self.feature_dim / 3)
        self.register_buffer('features', torch.rand(self.queue_size, self.feature_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('labels', torch.tensor([-1] * self.queue_size))
        print(f'Using queue shape: ({self.queue_size}, {self.feature_dim})')

    def store_keys(self, batch_features, batch_labels):
        batch_size = batch_features.size(0)
        batch_features.detach()
        batch_labels.detach()

        # update memory
        with torch.no_grad():
            store_indices = torch.arange(batch_size).cuda()
            store_indices += self.index
            store_indices = torch.fmod(store_indices, self.queue_size)
            store_indices = store_indices.long()
            self.features.index_copy_(0, store_indices, batch_features)
            self.labels.index_copy_(0, store_indices, batch_labels)
            self.index = (self.index + batch_size) % self.queue_size

    def get_queue(self):
        features = self.features.clone()
        labels = self.labels.clone()

        certain_flag = labels.ge(0)

        certain_features = features[certain_flag]
        certain_labels = labels[certain_flag]
        return certain_features, certain_labels

    def get_size(self):
        return self.queue_size, self.feature_dim
