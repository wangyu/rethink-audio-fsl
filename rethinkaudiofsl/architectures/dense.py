import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class FC1(nn.Module):
    def __init__(self, opt):
        super(FC1, self).__init__()

        self.last_relu = opt['userelu']
        self.use_bn = opt['usebn']
        self.last_dropout = (opt['dropout'] if ('dropout' in opt) else False)

        self.fc = nn.Linear(512, 2048, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(512)
        self.init_weight()

    def init_weight(self):
        if self.use_bn:
            init_bn(self.bn)
        init_layer(self.fc)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        if self.use_bn:
            x = self.bn(input)
            x = self.fc(x)
        else:
            x = self.fc(input)

        if self.last_relu:
            x = F.relu_(x)

        if self.last_dropout:
            x = F.dropout(x, p=0.5, training=self.training)

        out = x.view(x.size(0), -1)
        return out


def create_model(opt):
    return FC1(opt)