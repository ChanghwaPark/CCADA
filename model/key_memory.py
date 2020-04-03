import math

import torch
from torch import nn


class KeyMemory(nn.Module):
    def __init__(self, queue_size, feature_dim):
        super(KeyMemory, self).__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        # self.index = 0

        stdv = 1. / math.sqrt(self.feature_dim / 3)
        self.register_buffer('features', torch.rand(self.queue_size, self.feature_dim).mul_(2 * stdv).add_(-stdv))
        print(f'Using queue shape: ({self.queue_size}, {self.feature_dim})')

    def store_keys(self, batch_features, batch_indices):
        # batch_size = batch_features.size(0)
        batch_features.detach()
        batch_indices.detach()

        # update memory
        with torch.no_grad():
            # new_indices = torch.arange(batch_size).cuda()
            # new_indices += self.index
            # new_indices = torch.fmod(new_indices, self.queue_size)
            # new_indices = new_indices.long()
            self.features.index_copy_(0, batch_indices, batch_features)
            # self.index = (self.index + batch_size) % self.queue_size

    def get_queue(self, selected_indices):
        features = self.features.clone()
        selected_features = torch.index_select(features, dim=0, index=selected_indices)
        return selected_features

    def get_size(self):
        return self.queue_size, self.feature_dim
