import torch
import torch.nn as nn
import torch.nn.functional as F

import model.resnet as resnet
from model.utils import BatchNormDomain


class Model(nn.Module):
    def __init__(self, base_net='ResNet50', num_classes=31, bottleneck_dim=2048):
        super(Model, self).__init__()
        self.base_network = getattr(resnet, base_net)(num_domains=2, pretrained=True, frozen=['layer1'])  # TODO
        # self.base_network = getattr(resnet, base_net)(num_domains=2, pretrained=True, frozen=[])
        self.bn_domain = 0
        self.num_domains_bn = 2

        # self.bottleneck_layer = nn.Sequential(nn.Linear(self.base_network.out_dim, bottleneck_dim),
        #                                       BatchNormDomain(bottleneck_dim, self.num_domains_bn, nn.BatchNorm1d),
        #                                       nn.ReLU(),
        #                                       nn.Dropout(0.5))
        self.bottleneck_layer = nn.Sequential(nn.Conv2d(self.base_network.out_dim, bottleneck_dim,
                                                        kernel_size=1, stride=1, padding=0, bias=False),
                                              BatchNormDomain(bottleneck_dim, self.num_domains_bn, nn.BatchNorm2d),
                                              nn.ReLU(),
                                              nn.Dropout(0.5))

        # self.classifier_layer_list = [nn.Linear(bottleneck_dim, width),
        #                               nn.ReLU(),
        #                               nn.Dropout(0.5),
        #                               nn.Linear(width, num_classes)]
        # self.classifier_layer = nn.Sequential(nn.Linear(bottleneck_dim, num_classes))
        self.classifier_layer = nn.Sequential(nn.Conv2d(bottleneck_dim, num_classes,
                                                        kernel_size=1, stride=1, padding=0, bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

        # initialization
        # self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        # self.bottleneck_layer[0].bias.data.fill_(0.1)
        # self.classifier_layer[0].weight.data.normal_(0, 0.01)
        # self.classifier_layer[0].bias.data.fill_(0.0)
        nn.init.kaiming_uniform_(self.bottleneck_layer[0].weight, a=0.005)
        nn.init.kaiming_uniform_(self.classifier_layer[0].weight, a=0.01)

        # collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1},
                               {"params": self.bottleneck_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1}]

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, inputs):
        end_points = {}
        base_global_features, base_local_features = self.base_network(inputs)
        local_features = self.bottleneck_layer(base_local_features)
        end_points['features'] = local_features

        local_logits = self.classifier_layer(local_features)
        end_points['local_logits'] = local_logits

        logits = self.avgpool(local_logits)
        logits = logits.view(logits.size(0), -1)
        end_points['logits'] = logits

        # outputs 'tensor'
        predictions = torch.max(logits, 1)[1]
        end_points['predictions'] = predictions

        # outputs 'tensor'
        # confidences = [F.softmax(el, dim=0)[i].item() for i, el in zip(predictions, logits)]
        confidences = torch.max(F.softmax(logits, dim=1), 1)[0]
        end_points['confidences'] = confidences

        return end_points

    def get_parameter_list(self):
        return self.parameter_list
