from typing import Optional, Sequence, Tuple, Union
import torch
from escnn import gspaces, nn
from monai.networks.layers.convutils import (
    same_padding,
)

import math
from typing import Tuple

import torch
import torch.nn.functional as F


class RotationModule:
    def __init__(self, group, spatial_dims, pad_value=0.0, eps=1e-6):
        if group not in ("so2", "so3"):
            raise ValueError(f"Group must be one of ('so2', 'so3'), got {group}")

        self.group = group
        self.spatial_dims = spatial_dims
        self.eps = eps
        self.pad_value = pad_value

    def compute_rotation_matrix(self, pose):
        if self.group == "so2":
            return get_rotation_matrix_so2(pose, self.spatial_dims)

        return get_rotation_matrix_so3(pose, eps=self.eps)

    def __call__(self, img, pose, R=None):
        if R is None:
            R = self.compute_rotation_matrix(pose)

        if self.spatial_dims == 3:
            # assume data comes in -Z -Y X format, to match `escnn`.
            # the rotation matrix comes in XYZ form, so I transform it
            # to X -Y -Z
            R[:, -2:, :] = R[:, -2:, :] * -1
            R[:, :, -2:] = R[:, :, -2:] * -1

        if self.spatial_dims == 2:
            # affine_grid and grid_sample rotate the grid, not the signal.
            # to rotate the signal by alpha, we want to rotate the grid
            # by -alpha, so we make that correction here.
            R[:, -1, :] = R[:, -1, :] * -1
            R[:, :, -1] = R[:, :, -1] * -1

        # add a displacement vector of zeros to the rotation matrix
        disp = torch.tensor(0).expand(len(img), self.spatial_dims, 1).type_as(img)
        A = torch.cat((R, disp), dim=2)

        grid = F.affine_grid(A, img.size(), align_corners=False).type_as(img)

        y = F.grid_sample(img - self.pad_value, grid, align_corners=False)
        return y + self.pad_value


def get_rotation_matrix_so2(pose, spatial_dims):
    """Computes a (batch of) rotation matrix of the SO(2) group, in either 2d or 3d. In 3d,
    rotation is assumed to be about the Z axis (first axis).

    Parameters
    ----------
    pose: torch.Tensor
        A (batch of) equivariant 2d vector(s), of the form (cos(theta), sin(theta))

    spatial_dims: int
        Indicates whether it's 2d or 3d

    Returns
    -------
    The (batch of) rotation matrix
    """
    if len(pose.shape) != 2:
        v = pose.unsqueeze(0)
    else:
        v = pose

    if spatial_dims == 2:
        R = torch.stack(
            (
                torch.stack((v[:, 0], -v[:, 1]), dim=1),
                torch.stack((v[:, 1], v[:, 0]), dim=1),
            ),
            dim=1,
        )
    else:
        zeros = torch.tensor(0).type_as(v).expand(len(v))
        ones = torch.tensor(1).type_as(v).expand(len(v))

        R = torch.stack(
            (
                torch.stack((v[:, 0], -v[:, 1], zeros), dim=1),
                torch.stack((v[:, 1], v[:, 0], zeros), dim=1),
                torch.stack((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )

    if len(pose.shape) != 2:
        return R.squeeze(0)
    return R


def get_rotation_matrix_so3(z, eps=1e-6):
    """Computes a (batch of) rotation matrix of the SO(3) group.

    Parameters
    ----------
    z: torch.Tensor
        A batch of pairs of equivariant vectors, from which a rotation matrix
        is inferred

    eps: float
        Precision

    Returns
    -------
    The (batch of) rotation matrix
    """

    # produce  unit vector
    v1 = z[:, 0, :]
    u1 = v1 / (v1.norm(dim=1, keepdim=True) + eps)

    # produce a second unit vector, orthogonal to the first one
    v2 = z[:, 1, :]
    v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
    u2 = v2 / (v2.norm(dim=1, keepdim=True) + eps)

    # produce a third orthogonal vector, as the cross product of the first two
    u3 = torch.cross(u1, u2)
    # rot = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
    rot = torch.stack([u1, u2, u3], dim=1)

    return rot


def make_block(
    in_type,
    out_channels,
    stride,
    kernel_size,
    padding,
    spatial_dims,
    num_res_units,
    padding_mode="zeros",
    bias=True,
    batch_norm=True,
    activation=True,
    last_conv=False,
    out_vector_channels=None,
):
    if num_res_units > 0 and not last_conv:
        return ResBlock(
            spatial_dims=spatial_dims,
            in_type=in_type,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            subunits=num_res_units,
            bias=bias,
        )
    if padding is None:
        padding = same_padding(kernel_size)

    return Convolution(
        spatial_dims=spatial_dims,
        in_type=in_type,
        out_channels=out_channels,
        stride=stride,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=padding_mode,
        bias=bias,
        batch_norm=batch_norm and not last_conv,
        activation=activation and not last_conv,
        out_vector_channels=out_vector_channels,
    )

class ResBlock(nn.EquivariantModule):
    def __init__(
        self,
        spatial_dims,
        in_type,
        out_channels,
        stride,
        kernel_size,
        padding=None,
        padding_mode="zeros",
        subunits=2,
        bias=True,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_type = in_type

        if padding is None:
            padding = same_padding(kernel_size)

        subunits = max(1, subunits)
        conv = []

        prev_out_type = in_type
        sstride = stride
        spadding = padding
        for su in range(subunits):
            unit = Convolution(
                spatial_dims=spatial_dims,
                in_type=prev_out_type,
                out_channels=out_channels,
                stride=sstride,
                kernel_size=kernel_size,
                bias=bias,
                padding=spadding,
                padding_mode=padding_mode,
                batch_norm=True,
                activation=True,
            )

            sstride = 1
            spadding = same_padding(kernel_size)
            conv.append(unit)
            prev_out_type = unit.out_type
        self.conv = nn.SequentialModule(*conv)

        need_res_conv = (
            stride != 1
            or in_type != self.conv.out_type
            or (stride == 1 and padding < same_padding(kernel_size))
        )

        if need_res_conv:
            rkernel_size = kernel_size
            rpadding = padding

            # if only adapting number of channels a 1x1 kernel is used with no padding
            if stride == 1 and padding == same_padding(kernel_size):
                rkernel_size = 1
                rpadding = 0

            self.residual = Convolution(
                spatial_dims=spatial_dims,
                in_type=in_type,
                out_channels=out_channels,
                stride=stride,
                kernel_size=rkernel_size,
                bias=bias,
                padding=rpadding,
                padding_mode=padding_mode,
                batch_norm=False,
                activation=False,
            )
        else:
            self.residual = nn.IdentityModule(in_type)
        self.out_type = self.conv.out_type

    def forward(self, x):
        return self.residual(x) + self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        return input_shape
        

class Convolution(nn.EquivariantModule):
    def __init__(
        self,
        spatial_dims,
        in_type,
        out_channels,
        stride,
        kernel_size,
        bias=None,
        padding=None,
        padding_mode="zeros",
        batch_norm=True,
        activation=True,
        out_vector_channels=None,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_type = in_type
        gspace = in_type.gspace
        group = in_type.gspace.fibergroup
        out_vector_channels = (
            out_vector_channels if out_vector_channels is not None else out_channels
        )

        scalar_fields = nn.FieldType(gspace, out_channels * [gspace.trivial_repr])
        if type(group).__name__ in ("SO2", "SO3"):
            vector_fields = nn.FieldType(
                gspace, out_vector_channels * [gspace.irrep(1)]
            )
            out_type = scalar_fields + vector_fields
        else:
            vector_fields = []
            out_type = scalar_fields

        conv_class = nn.R3Conv if spatial_dims == 3 else nn.R2Conv
        conv = conv_class(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
        )

        if stride > 1:
            pool_class = (
                nn.PointwiseAvgPoolAntialiased2D
                if self.spatial_dims == 2
                else nn.PointwiseAvgPoolAntialiased3D
            )
            pool = pool_class(conv.out_type, sigma=0.33, stride=stride)
        else:
            pool = nn.IdentityModule(conv.out_type)
        if spatial_dims == 3 and batch_norm:
            batch_norm = get_batch_norm(scalar_fields, vector_fields, nn.IIDBatchNorm3d)
        elif spatial_dims == 2 and batch_norm:
            batch_norm = get_batch_norm(scalar_fields, vector_fields, nn.IIDBatchNorm2d)
        else:
            batch_norm = nn.IdentityModule(pool.out_type)

        if activation:
            activation = get_non_linearity(scalar_fields, vector_fields)
        else:
            activation = nn.IdentityModule(pool.out_type)

        self.net = nn.SequentialModule(conv, pool, batch_norm, activation)
        self.out_type = self.net.out_type

    def forward(self, x):
        return self.net(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        return input_shape


def get_non_linearity(scalar_fields, vector_fields):
    nonlinearity = nn.ReLU(scalar_fields)
    if len(vector_fields) > 0:
        out_type = scalar_fields + vector_fields
        norm_relu = nn.NormNonLinearity(vector_fields)
        nonlinearity = nn.MultipleModule(
            out_type,
            ["relu"] * len(scalar_fields) + ["norm"] * len(vector_fields),
            [(nonlinearity, "relu"), (norm_relu, "norm")],
        )

    return nonlinearity


def get_elu_non_linearity(scalar_fields, vector_fields):
    nonlinearity = nn.PointwiseNonLinearity(scalar_fields, "p_elu")

    if len(vector_fields) > 0:
        out_type = scalar_fields + vector_fields
        norm_nonlinearity = nn.NormNonLinearity(vector_fields)
        nonlinearity = nn.MultipleModule(
            out_type,
            ["elu"] * len(scalar_fields) + ["norm"] * len(vector_fields),
            [(nonlinearity, "elu"), (norm_nonlinearity, "norm")],
        )
    return nonlinearity


def get_batch_norm(scalar_fields, vector_fields, batch_norm_cls):
    batch_norm = batch_norm_cls(scalar_fields)
    if len(vector_fields) > 0:
        out_type = scalar_fields + vector_fields
        norm_batch_norm = nn.NormBatchNorm(vector_fields)
        batch_norm = nn.MultipleModule(
            out_type,
            ["bn"] * len(scalar_fields) + ["nbn"] * len(vector_fields),
            [(batch_norm, "bn"), (norm_batch_norm, "nbn")],
        )

    return batch_norm
