import torch
import math
from torch import nn
import torch.nn.functional as F


def bn_conv1d(in_planes, out_planes, kernel_size, dilated, bias):
    return nn.Sequential(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm1d(out_planes),
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dilated=1, bias=False, k1_sz=3, k2_sz=3):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = bn_conv1d(inplanes, planes, k1_sz, dilated, bias)
        self.conv2 = bn_conv1d(planes, planes, k2_sz, dilated, bias)
        if inplanes != planes:
            self.projection = bn_conv1d(inplanes, planes, 1, 1, bias)

    def forward(self, x):
        residual = x
        out = F.elu(self.conv1(x))
        out = self.conv2(out)
        if self.inplanes != self.planes:
            residual = self.projection(residual)
        out += residual
        out = F.elu(out)

        return out


class OnehotMSA(nn.Module):
    def __init__(self, planes, inplanes=23, onehot=False):
        super(OnehotMSA, self).__init__()
        self.emb = nn.Embedding(inplanes, planes)
        if onehot:
            assert planes == inplanes
            self.emb.weight.data = torch.eye(inplanes)
            self.emb.weight.requires_grad = False

    def forward(self, x):
        """
        x: (*, L)
        return: (*, C, L)
        """
        return self.emb(x).transpose(-1, -2)
