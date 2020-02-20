import math

import torch
from torch import nn


class KeyMemory(nn.Module):
    def __init__(self, queue_size, n_rkhs, n_dims):
        super(KeyMemory, self).__init__()
        self.queue_size = queue_size
        self.n_rkhs = n_rkhs
        self.n_dims = n_dims
        self.index = 0

        stdv = 1. / math.sqrt(self.n_rkhs * self.n_dims * self.n_dims / 3)
        self.register_buffer(
            'features', torch.rand(self.queue_size, self.n_rkhs, self.n_dims, self.n_dims).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('labels', torch.tensor([-1] * self.queue_size))
        print('Using queue shape: ({},{},{},{})'.format(self.queue_size, self.n_rkhs, self.n_dims, self.n_dims))

    def store_keys(self, batch_features, batch_labels):
        batch_size = batch_features.size(0)
        batch_features.detach()
        batch_labels.detach()

        # update memory
        with torch.no_grad():
            new_indices = torch.arange(batch_size).cuda()
            new_indices += self.index
            new_indices = torch.fmod(new_indices, self.queue_size)
            new_indices = new_indices.long()
            self.features.index_copy_(0, new_indices, batch_features)
            self.labels.index_copy_(0, new_indices, batch_labels)
            self.index = (self.index + batch_size) % self.queue_size

    def get_queue(self):
        features = self.features.clone()
        labels = self.labels.clone()
        return features, labels
