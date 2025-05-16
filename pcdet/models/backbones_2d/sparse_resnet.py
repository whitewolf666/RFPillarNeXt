import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from spconv.pytorch import SparseModule, SparseSequential, SparseConv2d, SparseConvTensor

from .conv import BasicBlock, ConvBlock
from .sparse_conv import SparseConvBlock, SparseBasicBlock


class SparseResNet(SparseModule):
    def __init__(self,
                 model_cfg,
                 **kwargs):
        super(SparseResNet, self).__init__()
        self.model_cfg = model_cfg
        layer_strides = self.model_cfg.DS_LAYER_STRIDES
        num_filters = self.model_cfg.DS_NUM_FILTERS
        layer_nums = self.model_cfg.LAYER_NUMS
        kernel_size = self.model_cfg.KERNEL_SIZES
        out_channels = self.model_cfg.OUT_CHANNELS
        input_channels = self.model_cfg.INPUT_CHANNELS

        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)

        in_filters = [input_channels, *num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(layer_nums):
            block = self._make_layer(
                in_filters[i],
                num_filters[i],
                kernel_size[i],
                layer_strides[i],
                layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.mapping = SparseSequential(
            SparseConv2d(num_filters[-1],
                         out_channels, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.pre_conv = BasicBlock(out_channels)
        self.conv1x1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, bias=False, padding=0
        )
        self.weight = nn.Parameter(torch.randn(out_channels, out_channels, 3, 3))
        self.post_conv = ConvBlock(out_channels * 6, out_channels, kernel_size=1, stride=1)
        self.num_bev_features = out_channels


    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes, kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return SparseSequential(*layers)

    def forward(self, data_dict):
        pillar_features = data_dict['pillar_features']
        coords = data_dict['voxel_coords']
        input_shape = data_dict['grid_size']
        batch_size = len(torch.unique(coords[:, 0]))
        x = SparseConvTensor(
            pillar_features, coords, input_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.mapping(x)
        x_dense = x.dense()
        if x_dense.requires_grad:
            out = cp.checkpoint(self._forward, x_dense)
        else:
            out = self._forward(x_dense)
        data_dict['spatial_features_2d'] = out
        return data_dict

    def _forward(self, x):
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=18, dilation=18)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        return x