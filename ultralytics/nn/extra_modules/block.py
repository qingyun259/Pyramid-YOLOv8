import torch
import torch.nn as nn
from ..modules.conv import Conv
from ..modules.block import *


__all__ = ['C2f_Faster', 'Partial_conv3']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

######################################## DyHead begin ########################################
try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError:
    pass


######################################## C2f-Faster begin ########################################

from timm.models.layers import DropPath

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=3, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - 2 * self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_conv3, self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.conv2(self.conv1(x1))
        x2 = self.partial_conv3(x2)
        x = torch.cat((x1, x2, x3), 1)
        return x

class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))

######################################## C2f-Faster end ########################################
