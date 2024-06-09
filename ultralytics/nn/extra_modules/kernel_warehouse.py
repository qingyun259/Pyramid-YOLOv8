import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial
from ..modules.conv import Conv, autopad

__all__ = [ 'Warehouse_Manager']

def parse(x, n):
    if isinstance(x, collections.abc.Iterable):
        if len(x) == 1:
            return list(repeat(x[0], n))
        elif len(x) == n:
            return x
        else:
            raise ValueError('length of x should be 1 or n')
    else:
        return list(repeat(x, n))

class Warehouse_Manager(nn.Module):
    def __init__(self, reduction=0.0625, cell_num_ratio=1, cell_inplane_ratio=1,
                 cell_outplane_ratio=1, sharing_range=(), nonlocal_basis_ratio=1,
                 norm_layer=nn.BatchNorm1d, spatial_partition=True):
        """
        Create a Kernel Warehouse manager for a network.
        Args:
            reduction (float or tuple): reduction ratio for hidden plane
            cell_num_ratio (float or tuple): number of kernel cells in warehouse / number of kernel cells divided
                        from convolutional layers, set cell_num_ratio >= max(cell_inplane_ratio, cell_outplane_ratio)
                        for applying temperature initialization strategy properly
            cell_inplane_ratio (float or tuple): input channels of kernel cells / the greatest common divisor for
                        input channels of convolutional layers
            cell_outplane_ratio (float or tuple): input channels of kernel cells / the greatest common divisor for
                        output channels of convolutional layers
            sharing_range (tuple): range of warehouse sharing.
                        For example, if the input is ["layer", "conv"], the convolutional layer "stageA_layerB_convC"
                        will be assigned to the warehouse "stageA_layer_conv"
            nonlocal_basis_ratio (float or tuple): reduction ratio for mapping kernel cells belongs to other layers
                        into fewer kernel cells in the attention module of a layer to reduce parameters, enabled if
                        nonlocal_basis_ratio < 1.
            spatial_partition (bool or tuple): If ``True``, splits kernels into cells along spatial dimension.
        """
        super(Warehouse_Manager, self).__init__()
        self.sharing_range = sharing_range
        self.warehouse_list = {}
        self.reduction = reduction
        self.spatial_partition = spatial_partition
        self.cell_num_ratio = cell_num_ratio
        self.cell_outplane_ratio = cell_outplane_ratio
        self.cell_inplane_ratio = cell_inplane_ratio
        self.norm_layer = norm_layer
        self.nonlocal_basis_ratio = nonlocal_basis_ratio
        self.weights = nn.ParameterList()

    def fuse_warehouse_name(self, warehouse_name):
        fused_names = []
        for sub_name in warehouse_name.split('_'):
            match_name = sub_name
            for sharing_name in self.sharing_range:
                if str.startswith(match_name, sharing_name):
                    match_name = sharing_name
            fused_names.append(match_name)
        fused_names = '_'.join(fused_names)
        return fused_names

    def reserve(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                bias=True, warehouse_name='default', enabled=True, layer_type='conv2d'):
        """
        Create a dynamic convolution layer without convolutional weights and record its information.
        Args:
            warehouse_name (str): the warehouse name of current layer
            enabled (bool): If ``False``, return a vanilla convolutional layer defined in pytorch.
            layer_type (str): 'conv1d', 'conv2d', 'conv3d' or 'linear'
        """
        kw_mapping = {'conv1d': KWConv1d, 'conv2d': KWConv2d, 'conv3d': KWConv3d, 'linear': KWLinear}
        org_mapping = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'conv3d': nn.Conv3d, 'linear': nn.Linear}

        if not enabled:
            layer_type = org_mapping[layer_type]
            if layer_type is nn.Linear:
                return layer_type(in_planes, out_planes, bias=bias)
            else:
                return layer_type(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
        else:
            layer_type = kw_mapping[layer_type]
            warehouse_name = self.fuse_warehouse_name(warehouse_name)
            weight_shape = [out_planes, in_planes // groups, *parse(kernel_size, layer_type.dimension)]

            if warehouse_name not in self.warehouse_list.keys():
                self.warehouse_list[warehouse_name] = []
            self.warehouse_list[warehouse_name].append(weight_shape)

            return layer_type(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias,
                              warehouse_id=int(list(self.warehouse_list.keys()).index(warehouse_name)),
                              warehouse_manager=self)

    def store(self):
        warehouse_names = list(self.warehouse_list.keys())
        self.reduction = parse(self.reduction, len(warehouse_names))
        self.spatial_partition = parse(self.spatial_partition, len(warehouse_names))
        self.cell_num_ratio = parse(self.cell_num_ratio, len(warehouse_names))
        self.cell_outplane_ratio = parse(self.cell_outplane_ratio, len(warehouse_names))
        self.cell_inplane_ratio = parse(self.cell_inplane_ratio, len(warehouse_names))

        for idx, warehouse_name in enumerate(self.warehouse_list.keys()):
            warehouse = self.warehouse_list[warehouse_name]
            dimension = len(warehouse[0]) - 2

            # Calculate the greatest common divisors
            out_plane_gcd, in_plane_gcd, kernel_size = warehouse[0][0], warehouse[0][1], warehouse[0][2:]
            for layer in warehouse:
                out_plane_gcd = math.gcd(out_plane_gcd, layer[0])
                in_plane_gcd = math.gcd(in_plane_gcd, layer[1])
                if not self.spatial_partition[idx]:
                    assert kernel_size == layer[2:]

            cell_in_plane = max(int(in_plane_gcd * self.cell_inplane_ratio[idx]), 1)
            cell_out_plane = max(int(out_plane_gcd * self.cell_outplane_ratio[idx]), 1)
            cell_kernel_size = parse(1, dimension) if self.spatial_partition[idx] else kernel_size

            # Calculate number of total mixtures to calculate for each stage
            num_total_mixtures = 0
            for layer in warehouse:
                groups_channel = int(layer[0] // cell_out_plane * layer[1] // cell_in_plane)
                groups_spatial = 1

                for d in range(dimension):
                    groups_spatial = int(groups_spatial * layer[2 + d] // cell_kernel_size[d])

                num_layer_mixtures = groups_spatial * groups_channel
                num_total_mixtures += num_layer_mixtures

            self.weights.append(nn.Parameter(torch.randn(
                max(int(num_total_mixtures * self.cell_num_ratio[idx]), 1),
                cell_out_plane, cell_in_plane, *cell_kernel_size), requires_grad=True))

    def allocate(self, network, _init_weights=partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')):
        num_warehouse = len(self.weights)
        end_idxs = [0] * num_warehouse

        for layer in network.modules():
            if isinstance(layer):
                warehouse_idx = layer.warehouse_id
                start_cell_idx = end_idxs[warehouse_idx]
                end_cell_idx = layer.init_attention(self.weights[warehouse_idx],
                                                    start_cell_idx,
                                                    self.reduction[warehouse_idx],
                                                    self.cell_num_ratio[warehouse_idx],
                                                    norm_layer=self.norm_layer,
                                                    nonlocal_basis_ratio=self.nonlocal_basis_ratio)
                _init_weights(self.weights[warehouse_idx][start_cell_idx:end_cell_idx].view(
                    -1, *self.weights[warehouse_idx].shape[2:]))
                end_idxs[warehouse_idx] = end_cell_idx

        for warehouse_idx in range(len(end_idxs)):
            assert end_idxs[warehouse_idx] == self.weights[warehouse_idx].shape[0]

    def take_cell(self, warehouse_idx):
        return self.weights[warehouse_idx]



def get_temperature(iteration, epoch, iter_per_epoch, temp_epoch=20, temp_init_value=30.0, temp_end=0.0):
    total_iter = iter_per_epoch * temp_epoch
    current_iter = iter_per_epoch * epoch + iteration
    temperature = temp_end + max(0, (temp_init_value - temp_end) * ((total_iter - current_iter) / max(1.0, total_iter)))
    return temperature