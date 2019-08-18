import torch.nn as nn
from model.utils import MLP
# from torchvision.models import r

class TowMLPHead(nn.Module):
    def __init__(self, layer_sizes, activation, bias=True, use_bn=True, drop_prob=None):
        super(TowMLPHead, self).__init__()
        self.head_fc = MLP(layer_sizes, activation, bias, use_bn, drop_prob)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.head_fc(x)

class ResBlockHead(nn.Module):
    def __init__(self, ):
        super(ResBlockHead, self).__init__()
        pass