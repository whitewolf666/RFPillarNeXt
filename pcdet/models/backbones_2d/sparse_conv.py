import torch.nn as nn
import spconv
from spconv.pytorch import SparseModule, SubMConv2d, SparseConv2d, SubMConv3d, SparseConv3d
from spconv.core import ConvAlgo

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseConvBlock(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True, bias=False):
        super(SparseConvBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = SubMConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1, bias=bias)
        else:
            self.conv = SparseConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                     padding=kernel_size // 2, stride=stride, bias=bias)

        self.norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))
        return out

class SparseBasicBlock(SparseModule):
    def __init__(self, channels, kernel_size):
        super(SparseBasicBlock, self).__init__()
        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1)
        self.conv2 = SubMConv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                                stride=1, bias=False, algo=ConvAlgo.Native)
        self.norm2 = nn.BatchNorm1d(channels, eps=0.001, momentum=0.01)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out