from typing import Optional, Sequence, Tuple, Union
import torch
from escnn import gspaces, nn
from monai.networks.layers.convutils import (
    same_padding,
)

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
