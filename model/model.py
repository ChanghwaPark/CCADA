import torch
import torch.nn as nn
import torch.nn.functional as F

import model.resnet as resnet
from model.utils import BatchNormDomain, initialize_layer


class Model(nn.Module):
    def __init__(self, base_net='ResNet50', num_classes=31, contrast_dim=128, frozen_layer='layer1'):
        super(Model, self).__init__()
        self.contrast_dim = contrast_dim
        self.bn_domain = 0
        self.num_domains_bn = 2

        # base network
        self.base_network = getattr(resnet, base_net)(num_domains=2, pretrained=True, frozen=[frozen_layer])
        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 1}]

        # contrast loss head
        self.contrast_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_network.out_dim, self.contrast_dim),
            BatchNormDomain(self.contrast_dim, self.num_domains_bn, nn.BatchNorm1d),
        )
        initialize_layer(self.contrast_layer)
        self.parameter_list += [{"params": self.contrast_layer.parameters(), "lr": 10}]

        # self.contrast_layer = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     BatchNormDomain(bottleneck_dim, self.num_domains_bn, nn.BatchNorm1d),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(bottleneck_dim, 128),
        #     BatchNormDomain(128, self.num_domains_bn, nn.BatchNorm1d)
        # )

        # self.classifier_layer_list = [nn.Linear(bottleneck_dim, width),
        #                               nn.ReLU(),
        #                               nn.Dropout(0.5),
        #                               nn.Linear(width, num_classes)]
        # classification head
        self.classifier_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_network.out_dim, num_classes)
        )
        initialize_layer(self.classifier_layer)
        self.parameter_list += [{"params": self.classifier_layer.parameters(), "lr": 10}]

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, inputs):
        end_points = {}
        features = self.base_network(inputs)
        features = F.normalize(features, p=2, dim=1)
        end_points['features'] = features

        # contrast loss head
        contrast_features = self.contrast_layer(features)
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        end_points['contrast_features'] = contrast_features

        logits = self.classifier_layer(features)
        end_points['logits'] = logits

        confidences, predictions = torch.max(F.softmax(logits, dim=1), 1)
        end_points['predictions'] = predictions
        end_points['confidences'] = confidences

        return end_points

    def get_parameter_list(self):
        return self.parameter_list
