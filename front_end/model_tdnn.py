""" TDNN-based speaker embedding network """

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, feat_dim=40, filters='512-512-512-512-1500', kernel_sizes='5-3-3-1-1', dilations='1-2-3-1-1',
                 pooling='stats', embedding_dims='512-512', n_class=5000, output_act='softmax'):
        super(TDNN, self).__init__()

        self.feat_dim = feat_dim
        self.filters = [int(filter_) for filter_ in filters.split('-')]
        self.kernel_sizes = [int(kernel_size_) for kernel_size_ in kernel_sizes.split('-')]
        self.dilations = [int(dilation_) for dilation_ in dilations.split('-')]
        assert len(self.filters) == len(self.kernel_sizes) == len(self.dilations), \
            'Unequal length of filters, kernel_sizes, or dilation rates!'

        self.pooling = pooling
        self.embedding_dims = [int(embedding_dim) for embedding_dim in embedding_dims.split('-')]
        self.n_class = n_class
        self.output_act = output_act

        # Create layers
        self.conv_layers = self.create_conv_layers(input_dim=self.feat_dim)
        self.pooling_layer, pooling_out_dim = self.create_pooling_layer(input_dim=self.filters[-1])
        self.emb_layers = self.create_emb_layers(input_dim=pooling_out_dim)
        self.output_layer = self.create_output_layer(input_dim=self.embedding_dims[-1])

    def create_conv_layers(self, input_dim=40):
        conv_layers = OrderedDict()

        conv_layers['conv0'] = conv1d_bn_relu_layer(input_dim, self.filters[0], self.kernel_sizes[0],
                                                    dilation=self.dilations[0])
        for i in range(1, len(self.filters)):
            conv_layers[f'conv{i}'] = conv1d_bn_relu_layer(self.filters[i - 1], self.filters[i], self.kernel_sizes[i],
                                                           dilation=self.dilations[i])

        return nn.Sequential(conv_layers)

    def create_pooling_layer(self, input_dim=1500):
        if self.pooling.startswith('attention'):
            _, hidden_nodes, heads = self.pooling.split('-')  # attention-500-1
            pooling_layer = AttentivePoolingLayer(in_nodes=input_dim, hidden_nodes=int(hidden_nodes),
                                                  heads=int(heads))
            pooling_out_dim = self.filters[-1] * 2 * heads
        elif self.pooling == 'stats':
            pooling_layer = StatsPoolingLayer()
            pooling_out_dim = self.filters[-1] * 2
        else:
            raise NotImplementedError

        return pooling_layer, pooling_out_dim

    def create_emb_layers(self, input_dim=300):
        emb_layers = OrderedDict()

        if len(self.embedding_dims) > 1:
            emb_layers['emb0'] = linear_bn_relu_layer(input_dim, self.embedding_dims[0])
            input_dim = self.embedding_dims[0]

            for i in range(1, len(self.embedding_dims) - 1):
                emb_layers[f'emb{i}'] = linear_bn_relu_layer(input_dim, self.embedding_dims[i])
                input_dim = self.embedding_dims[i]

        emb_layers[f'emb{len(self.embedding_dims) - 1}'] = nn.Sequential(
            OrderedDict([('linear', nn.Linear(input_dim, self.embedding_dims[-1], bias=False)),
                         ('bn', nn.BatchNorm1d(self.embedding_dims[-1], momentum=0.1))]))

        return nn.Sequential(emb_layers)

    def create_output_layer(self, input_dim=512):
        if self.output_act.startswith('amsoftmax'):  # 'amsoftmax-0.25-30'
            _, m, s = self.output_act.split('-')
            output_layer = AMSoftmaxLayer(in_nodes=input_dim, n_class=self.n_class, m=float(m), s=float(s))
        elif self.output_act.startswith('aamsoftmax'):
            _, m, s = self.output_act.split('-')
            output_layer = AAMSoftmaxLayer(in_nodes=input_dim, n_class=self.n_class, m=float(m), s=float(s))
        elif self.output_act == 'softmax':
            output_layer = SoftmaxLayer(in_nodes=input_dim, n_class=self.n_class)
        else:
            raise NotImplementedError

        return output_layer

    def forward(self, x, label=None):
        x = self.conv_layers(x)
        x = self.pooling_layer(x)
        x = self.emb_layers(x)

        return self.output_layer(x) if self.output_act == 'softmax' else self.output_layer(x, label)


class TDNNPreTrain(TDNN):
    def __init__(self, base_model, **kwargs):
        super(TDNNPreTrain, self).__init__(**kwargs)
        self.conv_layers = base_model.conv_layers
        self.pooling_layer = base_model.pooling_layer
        self.emb_layers = base_model.emb_layers
        self.output_layer = self.create_output_layer(input_dim=self.emb_layers[-1].get_submodule('linear').out_features)


class StatsPoolingLayer(nn.Module):
    def __init__(self, stats_type='standard', is_second_order_stat_only=False):
        super(StatsPoolingLayer, self).__init__()
        self.stats_type = stats_type
        self.is_second_order_stat_only = is_second_order_stat_only

    def forward(self, x):
        if self.stats_type == 'standard':
            std, mean = torch.std_mean(x, dim=-1, unbiased=False)
            return torch.cat((mean, std), dim=1) if not self.is_second_order_stat_only else std
        elif self.stats_type == 'rms':
            var, mean = torch.var_mean(x, dim=-1, unbiased=False)
            rms = torch.sqrt(torch.square(mean) + var)
            return torch.cat((mean, rms), dim=1) if not self.is_second_order_stat_only else rms
        else:
            raise NotImplementedError


class AttentivePoolingLayer(nn.Module):
    def __init__(self, in_nodes=1500, hidden_nodes=500, heads=1):
        """ attention_weight = softmax(tanh(x^T * W1) * W2)
            x: [B, C, L], L = No. of frames of an input sample
            W1: [C, D], D = hidden_node
            W2: [D, H], H = head
        """
        super(AttentivePoolingLayer, self).__init__()
        self.in_nodes = in_nodes
        self.heads = heads
        self.W1_layer = nn.Sequential(nn.Linear(in_nodes, hidden_nodes, bias=False), nn.Tanh())
        self.W2_layer = nn.Sequential(nn.Linear(hidden_nodes, heads, bias=False), nn.Softmax(dim=1))

    def forward(self, x):
        x_tr = x.transpose(1, 2)  # [B, L, C]
        x_tr = self.W1_layer(x_tr)  # [B, L, D]
        att_weight = self.W2_layer(x_tr)  # [B, L, H]

        mean = torch.einsum('ijk,ikl->ijl', x, att_weight)  # [B, C, H]
        var = F.relu(torch.einsum('ijk,ikl->ijl', torch.square(x), att_weight) - torch.square(mean))  # [B, C, H]
        mean = torch.reshape(mean, (-1, self.in_nodes * self.heads))  # [B, C * H]
        std = torch.sqrt(torch.reshape(var, (-1, self.in_nodes * self.heads)))  # [B, C * H]

        return torch.cat((mean, std), dim=1)


class SoftmaxLayer(nn.Module):
    def __init__(self, in_nodes=256, n_class=5000):
        super(SoftmaxLayer, self).__init__()
        self.layer = nn.Linear(in_nodes, n_class, bias=False)

    def forward(self, x):
        softmax_logits = self.layer(x)

        return softmax_logits, softmax_logits


class AMSoftmaxLayer(nn.Module):
    def __init__(self, in_nodes=256, n_class=5000, m=0.25, s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.randn(n_class, in_nodes), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, label):
        cos_theta = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1)).clamp(-1. + 1e-7, 1. - 1e-7)
        margin = torch.scatter(torch.zeros_like(cos_theta), 1, label.unsqueeze(1), self.m)
        amsoftmax_logits = self.s * (cos_theta - margin)

        return amsoftmax_logits, cos_theta  # return cos_theta as linear logits


class AAMSoftmaxLayer(nn.Module):
    def __init__(self, in_nodes=256, n_class=5000, m=0.2, s=30.):
        super(AAMSoftmaxLayer, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.randn(n_class, in_nodes), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.threshold = math.cos(math.pi - self.m)
        self.cos_margim = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label):
        cos_theta = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1)).clamp(-1. + 1e-7, 1. - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.mul(cos_theta, cos_theta)).clamp(1e-7, 1. - 1e-7))
        phi = cos_theta * math.cos(self.m) - sin_theta * math.sin(self.m)  # cos(theta + m)

        # When 0 <= theta + m < pi, then 0 <= theta < pi - m, and we have cos(theta) > cos(pi - m).
        # When theta + m >= pi or cos(theta) <= cos(pi - m), we use additive margin in the cosine domain.
        phi = torch.where(cos_theta > self.threshold, phi, cos_theta - self.cos_margim)

        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, label.view(-1, 1).long(), 1)
        aamsoftmax_logits = self.s * torch.where(one_hot.bool(), phi, cos_theta)
        # aamsoftmax_logits = self.s * ((one_hot * phi) + ((1.0 - one_hot) * cos_theta))

        return aamsoftmax_logits, cos_theta  # return cos_theta as linear logits


def conv1d_bn_relu_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    return nn.Sequential(OrderedDict([('conv', nn.Conv1d(in_channels, out_channels, kernel_size, stride=(stride,),
                                                         padding='same', dilation=(dilation,), bias=False)),
                                      ('bn', nn.BatchNorm1d(out_channels, momentum=0.1)),
                                      ('relu', nn.ReLU())]))


def linear_bn_relu_layer(in_nodes, out_nodes):
    return nn.Sequential(OrderedDict([('linear', nn.Linear(in_nodes, out_nodes, bias=False)),
                                      ('bn', nn.BatchNorm1d(out_nodes, momentum=0.1)),
                                      ('relu', nn.ReLU())]))
