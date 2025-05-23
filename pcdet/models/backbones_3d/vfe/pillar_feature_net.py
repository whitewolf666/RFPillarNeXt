import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_scatter
from functools import reduce

class PFNLayer(nn.Module):
    """
    Pillar Feature Net Layer.
    The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
    used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
    :param in_channels: <int>. Number of input channels.
    :param out_channels: <int>. Number of output channels.
    :param last_layer: <bool>. If last_layer, there is no concatenation of features.
    """

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs, unq_inv):
        torch.backends.cudnn.enabled = False
        x = self.linear(inputs)
        x = self.norm(x)
        x = F.relu(x)
        torch.backends.cudnn.enabled = True

        # max pooling
        feat_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        x_max = feat_max[unq_inv]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max], dim=1)
            return x_concatenated


class PillarNet(nn.Module):
    """
    PillarNet.
    The network performs dynamic pillar scatter that convert point cloud into pillar representation
    and extract pillar features

    Reference:
    PointPillars: Fast Encoders for Object Detection from Point Clouds (https://arxiv.org/abs/1812.05784)
    End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds (https://arxiv.org/abs/1910.06528)

    Args:
        num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        point_cloud_range: (<float>: 6). Point cloud range, only utilize x and y min.
    """

    def __init__(self,
                 num_input_features,
                 voxel_size,
                 grid_size,
                 point_cloud_range):
        super().__init__()
        self.voxel_size = np.array(voxel_size)
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)

    def forward(self, points):
        """
        Args:
            points: torch.Tensor of size (N, d), format: batch_id, x, y, z, feat1, ...
        """
        device = points.device
        dtype = points.dtype

        voxel_size = torch.from_numpy(
            self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.point_cloud_range).type_as(points).to(device)

        points_coords = (
            points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)   # x, y, z

        mask = reduce(torch.logical_and, (points_coords[:, 0] >= 0,
                                          points_coords[:, 0] < self.grid_size[0],
                                          points_coords[:, 1] >= 0,
                                          points_coords[:, 1] < self.grid_size[1]))

        points = points[mask]
        points_coords = points_coords[mask]

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()

        points_mean_scatter = torch_scatter.scatter_mean(
            points[:, 1:4], unq_inv, dim=0)

        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]

        # Find distance of x, y, and z from pillar center
        f_center = points[:, 1:3] - (points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0) +
                                     voxel_size[:2].unsqueeze(0) / 2 + pc_range[:2].unsqueeze(0))

        # Combine together feature decorations
        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)

        return features, unq[:, [0, 2, 1]], unq_inv, self.grid_size[[1, 0]]


class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        model_cfg,
        num_point_features,
        grid_size,
        voxel_size,
        point_cloud_range,
        **kwargs
    ):
        super().__init__()
        self.model_cfg = model_cfg
        num_filters = self.model_cfg.NUM_FILTERS
        norm_cfg = self.model_cfg.NORM_CFG
        num_point_features += 5

        # Create PillarFeatureNet layers
        self.num_filters = [num_point_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(self.num_filters) - 1):
            in_filters = self.num_filters[i]
            out_filters = self.num_filters[i + 1]
            if i < len(self.num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.voxelization = PillarNet(num_point_features, voxel_size, grid_size, point_cloud_range)

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        features, coords, unq_inv, grid_size = self.voxelization(points)
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)  # num_points, dim_feat

        feat_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        batch_dict['pillar_features'] = feat_max
        batch_dict['voxel_coords'] = coords
        batch_dict['grid_size'] = grid_size

        return batch_dict

    def get_output_feature_dim(self):
        return self.num_filters[-1]
