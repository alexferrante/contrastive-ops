import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import numpy as np
from typing import Dict, Tuple
import warnings
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from src.inv_model_utils import make_block, RotationModule, Convolution, create_mcvae_enc_so2_block

import escnn
from escnn import gspaces
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.simplelayers import Reshape

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def apply_scaled_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
class Encoder(torch.nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 num_input_channels: int, 
                 base_channel_size: int, 
                 apply_init_scale: bool,
                 variational: bool=False,
                 label_latent_dim: int=0,
                 BatchNorm = None,
                 act_fn: object = nn.GELU,
                 model=None,
                 width=64,
                 height=64,
                 scale_factor=0.1,
                 *args,
                 **kwargs):
        """
        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. 
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.variational = variational
        self.model = model
        self.latent_dim = latent_dim
        c_hid = base_channel_size
        if model is not None:
            if model == 'uhler':
                self.net = nn.Sequential(
                    nn.Conv2d(num_input_channels, c_hid, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(c_hid, c_hid * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(c_hid * 2, c_hid * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(c_hid * 4, c_hid * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(c_hid * 8, c_hid * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Flatten(),  # Image grid to single feature vector
                )
            elif model == 'test':
                self.net = nn.Sequential(
                    nn.Conv2d(num_input_channels, c_hid, kernel_size=4, stride=2, padding=1, bias=False),  # 96x96 => 48x48
                    nn.LayerNorm([c_hid, 48, 48]),
                    nn.GELU(),
                    nn.Conv2d(c_hid, c_hid * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 48x48 => 24x24
                    nn.LayerNorm([c_hid * 2, 24, 24]),
                    nn.GELU(),
                    nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=3, stride=2, padding=1, bias=False),  # 24x24 => 12x12
                    nn.LayerNorm([c_hid * 4, 12, 12]),
                    nn.GELU(),
                    nn.Conv2d(c_hid * 4, c_hid * 8, kernel_size=3, stride=2, padding=1, bias=False),  # 12x12 => 6x6
                    nn.LayerNorm([c_hid * 8, 6, 6]),
                    nn.GELU(),
                    nn.Conv2d(c_hid * 8, c_hid * 8, kernel_size=3, stride=2, padding=1, bias=False),  # 6x6 => 3x3
                    nn.LayerNorm([c_hid * 8, 3, 3]),
                    nn.GELU(),
                    nn.Flatten()  # Image grid to single feature vector
                )
            elif model in ["so2_multich", "multich", "so2_multich_pose"]:
                spatial_dims = 2
                num_res_units = 1

                channels = [16,32,64,128]
                strides = [1,1,2,2]
                kernel_sizes = [3]*4
                padding = [None]*4
                if "so2" in model:
                    self.gspace = (
                        gspaces.rot2dOnR2(N=-1, maximum_frequency=8)
                    )
                    n_out_vectors = 1
                else:
                    self.gspace = (
                        gspaces.trivialOnR2()
                    )
                    n_out_vectors = 0

                if model == "so2_multich_pose":
                    n_ch = 1
                else:
                    n_ch = num_input_channels
                self.in_type = escnn.nn.FieldType(self.gspace, n_ch*[self.gspace.trivial_repr])
    
                blocks = [ # (B, 48, input_spatial_dim, input_spatial_dim)
                    make_block(
                        in_type=self.in_type,
                        out_channels=channels[0],
                        stride=strides[0],
                        kernel_size=kernel_sizes[0],
                        padding=padding[0],
                        spatial_dims=spatial_dims,
                        num_res_units=num_res_units,
                        padding_mode="replicate",
                        batch_norm=BatchNorm,
                    )
                ]
                for c, s, k, p in zip(channels[1:], strides[1:], kernel_sizes[1:], padding[1:]):
                    blocks.append(make_block(blocks[-1].out_type, 
                                             out_channels=c, 
                                             stride=s, 
                                             kernel_size=k, 
                                             padding=p, 
                                             spatial_dims=spatial_dims, 
                                             num_res_units=num_res_units,
                                             batch_norm=BatchNorm))
    
                
                blocks.append(
                    make_block(
                        blocks[-1].out_type,
                        out_channels=latent_dim*2,  # encoder out size ? need to tweak for varational version 
                        stride=1,
                        kernel_size=1,
                        padding=0,
                        spatial_dims=spatial_dims,
                        num_res_units=num_res_units,
                        batch_norm=False,
                        activation=False,
                        last_conv=True,
                        out_vector_channels=n_out_vectors
                    )
                )
    
                # (B, 48, 69, 69)
                # (B, 96, 69, 69)
                # (B, 192, 35, 35)
                # (B, 384, 18, 18)
                # (B, 34, 18, 18)
    
                self.net = escnn.nn.SequentialModule(*blocks)
            elif "singlech" in model:
                spatial_dims = 2
                num_res_units = 1
                channels = [8, 16, 32, 64]
                strides = [1, 1, 2, 2]
                kernel_sizes = [3]*4
                padding = [None]*4
                
                if model == "so2_singlech":
                    self.gspace = (
                        gspaces.rot2dOnR2(N=-1, maximum_frequency=8)
                    )
                    n_out_vectors = 1
                else:
                    self.gspace = (
                        gspaces.trivialOnR2()
                    )
                    n_out_vectors = 0
    
                self.in_type = escnn.nn.FieldType(self.gspace, num_input_channels*[self.gspace.trivial_repr])
    
                blocks = [ # (B, 48, input_spatial_dim, input_spatial_dim)
                    make_block(
                        in_type=self.in_type,
                        out_channels=channels[0],
                        stride=strides[0],
                        kernel_size=kernel_sizes[0],
                        padding=padding[0],
                        spatial_dims=spatial_dims,
                        num_res_units=num_res_units,
                        padding_mode="replicate",
                    )
                ]
                for c, s, k, p in zip(channels[1:], strides[1:], kernel_sizes[1:], padding[1:]):
                    blocks.append(make_block(blocks[-1].out_type, 
                                             out_channels=c, 
                                             stride=s, 
                                             kernel_size=k, 
                                             padding=p, 
                                             spatial_dims=spatial_dims, 
                                             num_res_units=num_res_units))
    
                
                blocks.append(
                    make_block(
                        blocks[-1].out_type,
                        out_channels=latent_dim*2,
                        stride=1,
                        kernel_size=1,
                        padding=0,
                        spatial_dims=spatial_dims,
                        num_res_units=num_res_units,
                        batch_norm=False,
                        activation=False,
                        last_conv=True,
                        out_vector_channels=n_out_vectors
                    )
                )
    
                # (B, 48, 69, 69)
                # (B, 96, 69, 69)
                # (B, 192, 35, 35)
                # (B, 384, 18, 18)
                # (B, 34, 18, 18)
    
                self.net = escnn.nn.SequentialModule(*blocks)
            elif model == "so2_paper":
                
                self.gspace = (
                        gspaces.rot2dOnR2(N=-1, maximum_frequency=8)
                    )
                self.in_type = escnn.nn.FieldType(self.gspace, num_input_channels*[self.gspace.trivial_repr])
                n_out_vectors = 1
                
                blocks = [
                    Convolution(spatial_dims=2,
                                in_type=self.in_type,
                                out_channels=c_hid, 
                                kernel_size=3, 
                                padding=1, 
                                stride=2,
                                batch_norm=BatchNorm,
                                activation=True)]
                blocks.append(
                    Convolution(spatial_dims=2,
                                in_type=blocks[-1].out_type,
                                out_channels=c_hid, 
                                kernel_size=3, 
                                padding=1, 
                                stride=1,
                                batch_norm=BatchNorm,
                                activation=True))  # 64x64 => ? 32x32
                blocks.append(
                    Convolution(spatial_dims=2,
                                in_type=blocks[-1].out_type,
                                out_channels=2*c_hid, 
                                kernel_size=3, 
                                padding=1, 
                                stride=2,
                                batch_norm=BatchNorm,
                                activation=True)) # 64x64 => ? 32x32
                blocks.append(
                    Convolution(spatial_dims=2,
                                in_type=blocks[-1].out_type,
                                out_channels=2*c_hid, 
                                kernel_size=3, 
                                padding=1, 
                                stride=1,
                                batch_norm=BatchNorm,
                                activation=True))  # 64x64 => ? 32x32
                blocks.append(
                    Convolution(spatial_dims=2,
                                in_type=blocks[-1].out_type,
                                out_channels=2*c_hid, 
                                kernel_size=3, 
                                padding=1, 
                                stride=2,
                                batch_norm=False,
                                activation=True, 
                                out_vector_channels=1))  # 64x64 => ? 32x32
                self.net = escnn.nn.SequentialModule(*blocks)
            elif model in ["so2_direct_paper","so2_multich_pose_direct_paper"]:
                self.gspace = (
                        gspaces.rot2dOnR2(N=-1, maximum_frequency=8)
                    )
                if model == "so2_multich_pose_direct_paper":
                    nch=1
                else:
                    nch=num_input_channels
                self.in_type = escnn.nn.FieldType(self.gspace, nch*[self.gspace.trivial_repr])

                if self.latent_dim == 32:
                    kernels = [3]*5
                    paddings = [1]*5
                    strides = [2,1,2,1,2]
                    factors = [1,1,2,2,2] 
                elif self.latent_dim == 64:
                    kernels = [3]*6 
                    paddings = [1]*6
                    # [B, 130, 5, 5] pre-pool
                    strides = [2,1,2,1,2,2]
                    factors = [1,2,3,3,3,4]
                else:
                    raise NotImplementedError(f"Unsupported latent_dim: {self.latent_dim}")


                gspace = self.in_type.gspace
                in_type = self.in_type
                blocks = []
                
                for i, (kernel, padding, stride, factor) in enumerate(zip(kernels, paddings, strides, factors)):
                    if i == len(kernels) - 1:
                        out_ch = factor * c_hid
                        out_vch = (1)
                    else:
                        out_ch = factor * c_hid
                        out_vch = (out_ch)
                    
                    conv, act_fn = create_mcvae_enc_so2_block(gspace, in_type, out_ch, out_vch, kernel, padding, stride)
                    blocks.extend([conv, act_fn])
                    in_type = conv.out_type  
                    gspace = in_type.gspace 
                
                self.net = escnn.nn.SequentialModule(*blocks)
        else:
            if width == 96:
                self.net = nn.Sequential(
                    nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 96x96 => 48x48
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 48x48 => 24x24
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 24x24 => 12x12
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 12x12 => 6x6
                    act_fn(),
                    nn.Flatten(),  # Image grid to single feature vector
                )
            elif width == 64:
                self.net = nn.Sequential(
                    nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
                    act_fn(),
                    nn.Flatten(),  # Image grid to single feature vector
                )
        if apply_init_scale:
            apply_scaled_init(self.net)
        
        if self.variational:
            if model is not None:
                if model in ["multich", "so2_multich", "singlech", "so2_singlech", "so2_paper", \
                             "so2_multich_pose", "so2_multich_pose_direct_paper"]:
                    self.fc_mu = None
                    self.fc_log_var = None
                else:
                    input_size = c_hid * 8 * 3 * 3
                    self.fc_mu = nn.Linear(input_size, latent_dim)
                    self.fc_log_var = nn.Linear(input_size, latent_dim)
            else:
                if width == 96:
                    self.fc_mu = nn.Linear(2 * 6 * 6 * c_hid, latent_dim)
                    self.fc_log_var = nn.Linear(2 * 6 * 6 * c_hid, latent_dim)
                elif width == 64:
                    self.fc_mu = nn.Linear(2 * 8 * 8 * c_hid, latent_dim)
                    self.fc_log_var = nn.Linear(2 * 8 * 8 * c_hid, latent_dim)
                elif width == 69:
                    raise NotImplementedError
        else:
            self.net = nn.Sequential(self.net, nn.Linear(input_size, latent_dim))

        if self.fc_mu is not None and apply_init_scale:
            apply_scaled_init(self.fc_mu)
            apply_scaled_init(self.fc_log_var)

    def forward(self, x, **kwargs):
        if self.variational:
            if self.model is not None:
                if "multich" in self.model or "singlech" in self.model or "so2" in self.model:
                    if self.model in ["so2_multich_pose", "so2_multich_pose_direct_paper"]:
                        batch_size, num_channels, height, width = x.shape
                        channel_outputs = []
                        for i in range(num_channels):
                            ch = x[:, i:i+1, :, :]  #(B, 1, x, x)
                            ch = escnn.nn.GeometricTensor(ch, self.in_type)
                            out = self.net(ch)
                            channel_outputs.append(out.tensor)

                        y = torch.stack(channel_outputs, dim=1)
                        pool_dims = (3, 4)
                        y = y.mean(dim=pool_dims) # (B, 6, latent_dim+2)
                        y_pose = y[:, :, self.latent_dim*2:]

                        y = y.mean(dim=1)
                        y_embedding = y[:, :self.latent_dim*2] 
                        mu = y_embedding[:, :self.latent_dim]
                        log_var = y_embedding[:, self.latent_dim:]
                        
                        y_pose = y_pose[:, :, [1, 0]]
                        y_pose = y_pose / (torch.norm(y_pose, dim=-1, keepdim=True) + 1e-8)
                        return mu, log_var, y_pose
                    else:
                        x = escnn.nn.GeometricTensor(x, self.in_type)
                        y = self.net(x)
                        pool_dims = (2, 3)
                        y = y.tensor
                        y = y.mean(dim=pool_dims)
                        y_embedding = y[:, :self.latent_dim*2]
                        mu = y_embedding[:, :self.latent_dim]
                        log_var = y_embedding[:, self.latent_dim:]
                        if "so2" in self.model:
                            y_pose = y[:,self.latent_dim*2:]
                            y_pose = y_pose[:, [1, 0]]
                            y_pose = y_pose / (
                                    torch.norm(y_pose, dim=-1, keepdim=True) + 1e-8
                                ) # (B, 2)
                            return mu, log_var, y_pose
                        else:
                            return mu, log_var
            else:
                x = self.net(x)
                mu = self.fc_mu(x)
                log_var = self.fc_log_var(x)
                return mu, log_var
        else:
            x = self.net(x)
            return x

class Decoder(torch.nn.Module):
    def __init__(self, 
                 latent_dim: int, 
                 num_input_channels: int, 
                 base_channel_size: int, 
                 apply_init_scale: bool,
                 batch_latent_dim: int=0,
                 BatchNorm = None,
                 act_fn: object = nn.GELU,
                 model=None,
                 width=64,
                 height=64,
                 *args,
                 **kwargs):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct.
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        if model is not None:
            if model == 'uhler':
                print('using uhler decoder')
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim + batch_latent_dim, 2 * 6 * 6 * c_hid), 
                    act_fn(),
                    nn.Unflatten(1, (2 * c_hid, 6, 6)),
                )
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(c_hid * 8, c_hid * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(c_hid * 8, c_hid * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(c_hid * 4, c_hid * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(c_hid * 2, c_hid, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(c_hid),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(c_hid, num_input_channels, 4, 2, 1, bias=False),
                    nn.Tanh(),
            )
            elif model == 'test':
                print('using test decoder')
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim + batch_latent_dim, 8 * 3 * 3 * c_hid),
                    nn.LayerNorm(8 * 3 * 3 * c_hid),
                    act_fn(),
                    nn.Unflatten(1, (8 * c_hid, 3, 3)), 
                )
    
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(8 * c_hid, 4 * c_hid, kernel_size=4, padding=1, stride=2), # 3x3 => 6x6
                    nn.LayerNorm([4 * c_hid, 6, 6]),
                    act_fn(),
                    nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=4, padding=1, stride=2), # 6x6 => 12x12
                    nn.LayerNorm([2 * c_hid, 12, 12]),
                    act_fn(),
                    nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 12x12 => 24x24
                    nn.LayerNorm([c_hid, 24, 24]),
                    act_fn(),
                    nn.ConvTranspose2d(c_hid, c_hid // 2, kernel_size=5, output_padding=1, padding=2, stride=2), # 24x24 => 48x48, using a larger kernel
                    nn.LayerNorm([c_hid // 2, 48, 48]),
                    act_fn(),
                    nn.ConvTranspose2d(c_hid // 2, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 48x48 => 96x96
                    nn.Tanh(),
                )
            elif model in ["so2_multich", "multich", "so2_multich_pose"]:
                final_size = (18,18) # for LD 64 
    
                channels = [64, 32, 16, 6]
                strides = [2, 2, 2]
        
                decode_blocks = []
                for i, (s, c_in, c_out) in enumerate(
                    zip(strides, channels[:-1], channels[1:])
                ):
                    last_block = i + 1 == len(strides)
        
                    size = None if not last_block else (69, 69)
        
                    upsample = UpSample(
                        spatial_dims=2,
                        in_channels=c_in,
                        out_channels=c_in,
                        scale_factor=s,  # ignored if size isn't None, i.e. in the last bloc
                        size=size,
                        kernel_size=3,
                        pre_conv=None,
                        # choices inspired by this article:
                        # https://distill.pub/2016/deconv-checkerboard/
                        mode="nontrainable",
                        interp_mode="nearest",
                        align_corners=None,
                    )
        
                    res = ResidualUnit(
                        spatial_dims=2,
                        in_channels=c_in,
                        out_channels=c_out,
                        strides=1,
                        kernel_size=3,
                        act="relu",
                        norm="batch",
                        dropout=None,
                        subunits=1,
                        padding=1,
                    )
        
                    decode_blocks.append(nn.Sequential(upsample, res))
        
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim, channels[0] * int(np.product(final_size))),
                    Reshape(channels[0], *final_size),
                )
        
                self.net = nn.Sequential(
                    *decode_blocks,
                    nn.Identity(),
                )
            elif "singlech" in model:
                final_size = (18,18)
                channels = [64, 32, 16, 1]
                strides = [2,2,2]
    
                decode_blocks = []
                for i, (s, c_in, c_out) in enumerate(
                    zip(strides, channels[:-1], channels[1:])
                ):
                    last_block = i + 1 == len(strides)
        
                    size = None if not last_block else (69, 69)
        
                    upsample = UpSample(
                        spatial_dims=2,
                        in_channels=c_in,
                        out_channels=c_in,
                        scale_factor=s,  # ignored if size isn't None, i.e. in the last bloc
                        size=size,
                        kernel_size=3,
                        pre_conv=None,
                        # choices inspired by this article:
                        # https://distill.pub/2016/deconv-checkerboard/
                        mode="nontrainable",
                        interp_mode="nearest",
                        align_corners=None,
                    )
        
                    res = ResidualUnit(
                        spatial_dims=2,
                        in_channels=c_in,
                        out_channels=c_out,
                        strides=1,
                        kernel_size=3,
                        act="relu",
                        norm="batch",
                        dropout=None,
                        subunits=1,
                        padding=1,
                    )
        
                    decode_blocks.append(nn.Sequential(upsample, res))
        
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim, channels[0] * int(np.product(final_size))),
                    Reshape(channels[0], *final_size),
                )
        
                self.net = nn.Sequential(
                    *decode_blocks,
                    nn.Identity(),
                )
            elif model in ["so2_paper", "so2_direct_paper", "so2_multich_pose_direct_paper"]:
                if latent_dim == 64:
                    self.linear = nn.Sequential(
                        nn.Linear(latent_dim + batch_latent_dim, 2 * 8 * 8 * c_hid), 
                        act_fn(),
                        nn.Unflatten(1, (-1, 8, 8)),
                        )
                    self.net = nn.Sequential(
                        nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                        act_fn(),
                        nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                        act_fn(),
                        nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=0, stride=2),  # 16x16 => 34x34
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                        act_fn(),
                        nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=0, padding=0, stride=2),  # Adjusted for 34x34 => 69x69
                        nn.Tanh(),
                    )
                elif latent_dim == 128:  # keep the same for now
                    self.linear = nn.Sequential(
                        nn.Linear(latent_dim + batch_latent_dim, 2 * 8 * 8 * c_hid), 
                        act_fn(),
                        nn.Unflatten(1, (-1, 8, 8)),
                        )
                    self.net = nn.Sequential(
                        nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                        act_fn(),
                        nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                        act_fn(),
                        nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=0, stride=2),  # 16x16 => 34x34
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                        act_fn(),
                        nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=0, padding=0, stride=2),  # Adjusted for 34x34 => 69x69
                        nn.Tanh(),
                    )
        else:
            if width == 96:
                print('using width 96 decoder')
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim + batch_latent_dim, 2 * 6 * 6 * c_hid), 
                    act_fn(),
                    nn.Unflatten(1, (2 * c_hid, 6, 6)),
                )
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(c_hid, c_hid // 2, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
                    act_fn(),
                    nn.ConvTranspose2d(c_hid // 2, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 64x64 => 96x96
                    nn.Tanh(),
                )
            elif width == 64:
                print('using width 64 decoder')
                self.linear = nn.Sequential(
                    nn.Linear(latent_dim + batch_latent_dim, 2 * 8 * 8 * c_hid), 
                    act_fn(),
                    nn.Unflatten(1, (-1, 8, 8)),
                    )
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                    act_fn(),
                    nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
                    nn.Tanh(),
                )
        if apply_init_scale:
            apply_scaled_init(self.linear)
            apply_scaled_init(self.net)


    def forward(self, x, **kwargs):
        x = self.linear(x)
        x = self.net(x)
        return x
        
class BaseModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        optimizer_param: dict,
        latent_dim: int=32,
        base_channel_size: int=32,
        num_input_channels: int = 4,
        image_size: int = 64,
        act_fn=nn.GELU,
        *args,
        **kwargs,
        ):
        super().__init__()
        
        self.model_name = model_name
        self.num_input_channels = num_input_channels
        self.width = image_size
        self.height = image_size
        self.base_channel_size = base_channel_size
        self.latent_dim = latent_dim
        self.network_param = {'latent_dim': latent_dim, 'num_input_channels':num_input_channels,
                              'base_channel_size':base_channel_size, 'act_fn':act_fn}

        # Example input array needed for visualizing the graph of the network
        self.optimizer_param = optimizer_param
        

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_loss(self, x):
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        lr = self.optimizer_param['lr']
        if self.optimizer_param['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer_param['optimizer'] == 'SGD':
            momentum = self.optimizer_param['momentum']
            nesterov = self.optimizer_param['nesterov']
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss.detach())
        while True:  # Inside your training loop
            current_memory_allocated = torch.cuda.memory_allocated() / (1024.0 ** 3)  # Convert bytes to GB
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 3)  # Convert bytes to GB
            self.log_dict({'Current GPU Memory (GB)': current_memory_allocated, 'Max GPU Memory (GB)': max_memory_allocated})
            return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss.detach())

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss.detach())

class AEmodel(BaseModel):
    def __init__(self, 
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder_class(variational=False, **self.network_param)
        self.decoder = decoder_class(**self.network_param)
        self.example_input_array = torch.zeros(2, self.num_input_channels, self.width, self.height)

    def get_image_embedding(self, x):
        return self.encoder(x)

class VAEmodel(BaseModel):
    def __init__(self, 
                 step_size: int,
                 latent_dim = 64,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 *args, **kwargs):
        super().__init__(latent_dim=latent_dim, *args, **kwargs)
        
        # Initialize the cyclic weight scheduler
        self.kl_weight_scheduler = CyclicWeightScheduler(step_size=step_size)

        # for the gaussian likelihood
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))
        self.encoder = encoder_class(variational=True, **self.network_param)
        self.decoder = decoder_class(**self.network_param)
        self.example_input_array = torch.zeros(2, self.num_input_channels, self.width, self.height)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
    
    def forward(self, x):
        # encode x to get the mu and variance parameters
        mu, log_var = self.encoder(x)
        # sample z 
        log_var = torch.maximum(log_var, torch.tensor(-20)) #clipping to prevent going to -inf
        std = torch.exp(log_var / 2)
        z = self.sampling(mu, std)
        # decoded 
        x_hat = self.decoder(z)
        return x_hat, mu, std
    
    def get_image_embedding(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    @staticmethod
    def sampling(mu, std):
        q = Normal(mu, std)
        return q.rsample()

    @staticmethod
    def reconstruction_loss(sample: torch.Tensor,
                            mean: torch.Tensor, 
                            logscale: torch.Tensor, 
                            ):
        scale = torch.exp(logscale)
        dist = Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return -log_pxz.sum(dim=(1, 2, 3))

    def _generic_loss(self, x, x_hat, mu, std):
        # reconstruction probability
        recon_loss = self.reconstruction_loss(x, x_hat, self.log_scale)

        # kl
        kl = self.latent_kl_divergence(mu, std)
        kl_term_weight = self.kl_weight_scheduler.step()

        # elbo
        elbo = (kl_term_weight*kl + recon_loss)
        elbo = elbo.mean()
        self.log_dict({
            'elbo': elbo.detach(),
            'kl': kl.mean().detach(),
            'recon_loss': recon_loss.mean().detach(), 
            'kl_term_weight': kl_term_weight,
        })
        return elbo

    @staticmethod
    def latent_kl_divergence(variational_mean, 
                             variational_std, 
                             prior_mean=None,
                             prior_std=None) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and standard Gaussian prior.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(variational_mean)
        prior_std = torch.ones_like(variational_std)
        return kl(
            Normal(variational_mean, variational_std),
            Normal(prior_mean, prior_std),
        ).sum(dim=-1)
    
    def _get_loss(self, x):
        # get reconstruction, mu and std
        x_hat, mu, std = self.forward(x)
        elbo = self._generic_loss(x, x_hat, mu, std)
        return elbo

class ContrastiveVAEmodel(BaseModel):
    """
    Args:
    ----
        n_z_latent: Dimensionality of the background latent space.
        n_s_latent: Dimensionality of the salient latent space.
        wasserstein_penalty: Weight of the Wasserstein distance loss that further
            discourages shared variations from leaking into the salient latent space.
    """
        
    def __init__(self, 
                 n_z_latent: int = 32, 
                 n_s_latent: int = 32, 
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 step_size: float=2000,
                 ngene: int=None,
                 adjust_prior_s: bool=False,
                 adjust_prior_z: bool=False,
                 classify_s: bool=False,
                 classify_z: bool=False,
                 wasserstein_penalty: float = 0,
                 BatchNorm = None,
                 n_unique_batch: int = 34,
                 model = None,
                 batch_size: int=1024,
                 tc_penalty: float=1,
                 classification_weight: float=1,
                 scale_factor: float=0.1,
                 max_kl_weight: float=1,
                 batch_latent_dim: int=32,
                 reg_type: str=None,
                 total_steps: int=3000,
                 klscheduler: str='cyclic',
                 apply_init_scale: bool=True,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size = step_size
        self.ngene = ngene
        self.adjust_prior_s = adjust_prior_s
        self.adjust_prior_z = adjust_prior_z
        self.n_s_latent = n_s_latent
        self.n_z_latent = n_z_latent
        self.wasserstein_penalty = wasserstein_penalty
        self.BatchNorm = BatchNorm
        self.n_unique_batch = n_unique_batch
        self.model = model
        self.batch_size = batch_size
        self.tc_penalty = tc_penalty
        self.classify_s = classify_s
        self.classify_z = classify_z
        self.classification_weight = classification_weight
        self.scale_factor = scale_factor
        self.batch_latent_dim = batch_latent_dim
        self.reg_type = reg_type
        self.total_steps = total_steps
        self.klscheduler = klscheduler

        if n_s_latent != n_z_latent:
            warnings.warn('Target latent dim does not equal background latent dim')

        # Initialize the weight scheduler
        if self.klscheduler == 'cyclic':
            self.kl_weight_scheduler = CyclicWeightScheduler(step_size=self.step_size, max_weight=max_kl_weight)
        elif self.klscheduler == 'ramp':
            self.kl_weight_scheduler = KLRampScheduler(total_steps=self.total_steps, max_weight=max_kl_weight)

        # Background encoder

        self.coder_param = {'num_input_channels': self.num_input_channels, "scale_factor": self.scale_factor, 
                         'base_channel_size': self.base_channel_size, 'variational': True, 'width': self.width, 'height':self.height,
                          'BatchNorm': self.BatchNorm, 'n_unique_batch': self.n_unique_batch, 'model': self.model,
                           'apply_init_scale': apply_init_scale}
        self.z_encoder = encoder_class(
                                       latent_dim=self.n_z_latent,
                                       **self.coder_param,
                                       )
        # Salient encoder 
        self.s_encoder = encoder_class(
                                       latent_dim=self.n_s_latent,
                                       **self.coder_param,
                                       )
        
        # Decoder from latent variable to distribution parameters in data space.
        self.n_total_latent = self.n_z_latent + self.n_s_latent        
        self.decoder = decoder_class(
                                    latent_dim=self.n_total_latent,
                                    batch_latent_dim=self.batch_latent_dim,
                                    **self.coder_param,
                                    )
        
        if self.adjust_prior_z:
            self.zprior_embedding = torch.nn.Embedding(self.ngene, self.n_z_latent)
        if self.adjust_prior_s:
            self.sprior_embedding = torch.nn.Embedding(self.ngene, self.n_s_latent)
    
        # for the gaussian likelihood
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))

        # Example input array needed for visualizing the graph of the network
        # self.example_input_array = {'background': torch.zeros(2, self.num_input_channels, self.width, self.height),
        #                                 'target': torch.zeros(2, self.num_input_channels, self.width, self.height)}
        
        # if self.adjust_prior_s or self.adjust_prior_z:
        #     self.example_input_array['background_label'] = torch.zeros(2, dtype=torch.int32)
        #     self.example_input_array['target_label'] = torch.zeros(2, dtype=torch.int32)
        # if self.batch_latent_dim > 0:
        #     self.batch_embedding = torch.nn.Embedding(self.n_unique_batch, self.batch_latent_dim)
        #     self.example_input_array.update({'background_batch': torch.zeros(2, dtype=torch.int32),
        #                                     'target_batch': torch.zeros(2, dtype=torch.int32)})
        if model is not None and "so2" in model:
            self.rotation_module = RotationModule(
                "so2", 2, 0, 1e-8
            )
        else:
            self.rotation_module = None

        self.save_hyperparameters()

    def forward(self, background, target, **kwargs):
        background_label = kwargs.get('background_label')
        target_label = kwargs.get('target_label')
        prior_mu_background = {'zprior_m': None,  'sprior_m': None}
        prior_mu_target = {'zprior_m': None, 'sprior_m': None}
        
        if self.adjust_prior_s:
            prior_mu_background['sprior_m'] = self.sprior_embedding(background_label.int())
            prior_mu_target['sprior_m'] = self.sprior_embedding(target_label.int())

        if self.adjust_prior_z:
            prior_mu_background['zprior_m'] = self.zprior_embedding(background_label.int())
            prior_mu_target['zprior_m'] = self.zprior_embedding(target_label.int())

        inference_outputs = self.inference(background=background, target=target)
        background_batch = kwargs.get('background_batch')
        target_batch = kwargs.get('target_batch')
        generative_outputs = self.generative(inference_outputs['background'], 
                                             inference_outputs['target'],
                                             background_batch=background_batch,
                                             target_batch=target_batch)
        bg = generative_outputs['background']["px_m"]
        tg = generative_outputs['target']["px_m"]
        if self.rotation_module is not None:
            # use inference_outputs["background"]["zpose"] or ["spose"] for pose adjust 
            # somewhat arbitrarily choosing spose here...
            if self.model in ["so2_multich_pose","so2_multich_pose_direct_paper"]:
                bg_channels = []
                tg_channels = []
            
                for ch_idx in range(bg.shape[1]):
                    bg_ch = self.rotation_module(bg[:, ch_idx:ch_idx+1, ...], 
                                                 inference_outputs["background"]["spose"][:, ch_idx, ...])
                    bg_channels.append(bg_ch)
            
                    tg_ch = self.rotation_module(tg[:, ch_idx:ch_idx+1, ...], 
                                                 inference_outputs["target"]["spose"][:, ch_idx, ...])
                    tg_channels.append(tg_ch)
            
                # Stack the processed channels back along the channel dimension
                bg = torch.cat(bg_channels, dim=1)
                tg = torch.cat(tg_channels, dim=1)
            else:
                bg = self.rotation_module(bg, inference_outputs["background"]["spose"])
                tg = self.rotation_module(tg, inference_outputs["target"]["spose"])
        
        recon = {'bg':bg, 
                 "tg":tg}
            
        
        inference_outputs['background'].update(prior_mu_background)
        inference_outputs['target'].update(prior_mu_target)

        return recon, inference_outputs, generative_outputs
    
    def get_image_embedding(self, img, label=None):
        if self.model is None:
            qz_m, _ = self.z_encoder(img)
            qs_m, _ = self.s_encoder(img)
        else:
            qz_m, _, _ = self.z_encoder(img)
            qs_m, _, _ = self.s_encoder(img)
            
        return torch.cat((qs_m, qz_m), dim=1)

    def _generic_inference(self, x: torch.Tensor):
        if self.model is not None and "so2" in self.model:
            qz_m, qz_lv, zpose = self.z_encoder(x)
            qs_m, qs_lv, spose = self.s_encoder(x)
        else:
            qz_m, qz_lv = self.z_encoder(x)
            qs_m, qs_lv = self.s_encoder(x)
            zpose = None
            spose = None
        
        # Sample from latent distribution
        qz_lv = torch.maximum(qz_lv, torch.tensor(-20))  # Clipping to prevent going to -inf
        qs_lv = torch.maximum(qs_lv, torch.tensor(-20))  # Clipping to prevent going to -inf
        qz_s = torch.exp(qz_lv / 2)
        qs_s = torch.exp(qs_lv / 2)
        qz = Normal(qz_m, qz_s)
        qs = Normal(qs_m, qs_s)
        z = qz.rsample()
        s = qs.rsample()
    
        outputs = {
            "qz_m": qz_m,
            "qz_s": qz_s,
            "z": z,
            "qs_m": qs_m,
            "qs_s": qs_s,
            "s": s,
        }
    
        if zpose is not None:
            outputs["zpose"] = zpose
        if spose is not None:
            outputs["spose"] = spose
        return outputs

    def inference(
            self,
            background: torch.Tensor,
            target: torch.Tensor,
        ) -> Dict[str, Dict[str, torch.Tensor]]:
            background_batch_size = background.shape[0]
            target_batch_size = target.shape[0]
            inference_input = torch.cat([background, target], dim=0)
            outputs = self._generic_inference(x=inference_input)
            background_outputs, target_outputs = {}, {}
            for key in outputs.keys():
                if outputs[key] is not None:
                    background_tensor, target_tensor = torch.split(
                        outputs[key],
                        [background_batch_size, target_batch_size],
                        dim=0,
                    )
                else:
                    background_tensor, target_tensor = None, None
                background_outputs[key] = background_tensor
                target_outputs[key] = target_tensor
            background_outputs["s"] = torch.zeros_like(background_outputs["s"])
            return dict(background=background_outputs, target=target_outputs)
    
    def _generic_generative(self, 
                              z: torch.Tensor, 
                              s: torch.Tensor,
                              batch_embedding: torch.Tensor=None,):
        latent = torch.cat([z, s], dim=-1)
        if batch_embedding is not None:
            latent = torch.cat([latent, batch_embedding], dim=-1)
        px_m = self.decoder(latent)            
        return dict(px_m=px_m, px_s=self.log_scale)

    def generative(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        **kwargs,
        ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]
        
        generative_input = {}
        for key in ["z", "s"]:
            generative_input[key] = torch.cat(
                [background[key], target[key]], dim=batch_size_dim
            )

        # unused
        background_batch = kwargs.get("background_batch")
        target_batch = kwargs.get("target_batch")
        if background_batch is not None and target_batch is not None:
            generative_input["batch_embedding"] = torch.cat(
                [self.batch_embedding(background_batch), 
                 self.batch_embedding(target_batch)], dim=batch_size_dim
                )
        #
        
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        if outputs["px_m"] is not None:
            background_tensor, target_tensor = torch.split(outputs["px_m"],[background_batch_size, target_batch_size],dim=batch_size_dim,)
        else:
            background_tensor, target_tensor = None, None
        background_outputs["px_m"] = background_tensor
        target_outputs["px_m"] = target_tensor
        background_outputs["px_s"] = outputs["px_s"]
        target_outputs["px_s"] = outputs["px_s"]
        return dict(background=background_outputs, target=target_outputs)

    def _generic_loss(self, 
                      tensors: torch.Tensor,
                      inference_outputs: Dict[str, torch.Tensor], 
                      generative_outputs: Dict[str, torch.Tensor],
                    )-> Dict[str, torch.Tensor]:
        
        qz_m = inference_outputs["qz_m"]
        qz_s = inference_outputs["qz_s"]
        qs_m = inference_outputs["qs_m"]
        qs_s = inference_outputs["qs_s"]
        zprior_m = inference_outputs["zprior_m"]
        sprior_m = inference_outputs["sprior_m"]
        px_m = generative_outputs["px_m"]
        px_s = generative_outputs["px_s"]

        recon_loss = VAEmodel.reconstruction_loss(tensors, px_m, px_s)
        kl_z = VAEmodel.latent_kl_divergence(qz_m, qz_s, prior_mean=zprior_m)
        kl_s = VAEmodel.latent_kl_divergence(qs_m, qs_s, prior_mean=sprior_m)
        return dict(recon_loss=recon_loss, kl_z=kl_z, kl_s=kl_s)
    
    def compute_independent_loss(self, zb, zc):
        reg_type = self.reg_type
        if reg_type == "TC":
            return self.compute_tc(zb, zc)
        elif reg_type == "HSIC":
            return self.compute_HSIC(zb, zc)
        else:
            raise ValueError("reg_type should be TC or HSIC")
    
    @staticmethod
    def rbf_kernel(X, sigma=1.0):
        # Compute the pairwise squared Euclidean distances
        pairwise_dists = torch.cdist(X, X, p=2) ** 2
        # Apply the RBF kernel function
        values = torch.div(-pairwise_dists, (2 * sigma**2))
        return values.exp()
    
    @staticmethod
    def compute_HSIC(Z_b, Z_c):
        n = Z_b.shape[0]
        # Compute kernel matrices
        K = ContrastiveVAEmodel.rbf_kernel(Z_b)
        L = ContrastiveVAEmodel.rbf_kernel(Z_c)
        # print(K.shape, L.shape)
        # Implement the HSIC formula
        term1 = (1 / (n**2)) * torch.sum(K * L)
        term2 = (1 / (n**4)) * torch.sum(K) * torch.sum(L)
        term3 = (2 / (n**3)) * torch.sum(K @ L)
        HSIC_n = term1 + term2 - term3
        return HSIC_n * n
    
    @staticmethod
    def compute_tc(zb, zc):
        # Calculate the empirical means
        mean_zb = torch.mean(zb, dim=0)
        mean_zc = torch.mean(zc, dim=0)
        # Calculate the centered variables
        centered_zb = zb - mean_zb
        centered_zc = zc - mean_zc
        # Calculate the covariance matrix of the concatenated latent variables
        z_concat = torch.cat([centered_zb, centered_zc], dim=1)
        cov_matrix = torch.matmul(z_concat.T, z_concat) / z_concat.shape[0]
        # Calculate the covariance matrices for zb and zc individually
        cov_zb = torch.matmul(centered_zb.T, centered_zb) / centered_zb.shape[0]
        cov_zc = torch.matmul(centered_zc.T, centered_zc) / centered_zc.shape[0]
        # Calculate total correlation loss
        tc_loss = torch.logdet(cov_matrix) - (torch.logdet(cov_zb) + torch.logdet(cov_zc))
        # Multiply by the weighting factor
        return -tc_loss

    def _get_loss(self, 
             concat_tensors: Dict[str, Tuple[Dict[str, torch.Tensor], int]],
             ):  
        _, inference_outputs, generative_outputs = self.forward(**concat_tensors)            

        background_losses = self._generic_loss(
            concat_tensors["background"],
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            concat_tensors["target"],
            inference_outputs["target"],
            generative_outputs["target"],
        )
        recon_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_divergence_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_s = target_losses["kl_s"]

        wasserstein_loss = (
            torch.norm(inference_outputs["background"]["qs_m"], dim=-1)**2
            + torch.sum(inference_outputs["background"]["qs_s"]**2, dim=-1)
        )

        if self.reg_type is not None:
            zb = torch.concat([inference_outputs["target"]["qz_m"], inference_outputs["background"]["qz_m"]], axis=0)
            zs = torch.concat([inference_outputs["target"]["qs_m"], inference_outputs["background"]["qs_m"]], axis=0)
            tc_loss = self.compute_independent_loss(zb, zs)
        else:
            tc_loss = torch.zeros(1, device=self.device)

        kl_term_weight = self.kl_weight_scheduler.step()
        
        elbo = torch.mean(recon_loss + 
                          kl_term_weight * (kl_divergence_s + kl_divergence_z + 
                                            self.wasserstein_penalty * wasserstein_loss +
                                            self.tc_penalty * tc_loss))

        self.log_dict({
            'kl_divergence_z': kl_divergence_z.mean().detach(),
            'kl_divergence_s': kl_divergence_s.mean().detach(),
            'total_recon_loss': recon_loss.mean().detach(),
            'wasserstein_loss': wasserstein_loss.mean().detach(),
            'tc_loss': tc_loss.mean().detach(),
            # 'background_recon_loss': background_losses["recon_loss"].mean().detach(),
            # 'target_recon_loss': target_losses["recon_loss"].mean().detach(),
            'kl_term_weight': kl_term_weight,
        })
        return elbo
    
 
class CyclicWeightScheduler:
    def __init__(self, step_size, base_weight=0, max_weight=1):
        self.base_weight = base_weight
        self.max_weight = max_weight
        self.step_size = step_size
        self.cycle = 0
        self.step_count = 0

    def step(self):
        # Compute the current position in the cycle
        cycle_position = self.step_count / self.step_size

        if cycle_position <= 1:
            weight = self.base_weight + (self.max_weight - self.base_weight) * cycle_position
        else:
            weight = self.max_weight
            # weight = self.max_weight - (self.max_weight - self.base_weight) * (cycle_position - 1)

        self.step_count = (self.step_count + 1) % (self.step_size * 2)

        return weight
    
class KLRampScheduler:
    def __init__(self, start_weight=0, max_weight=1, total_steps=3000):
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.total_steps = total_steps
        self.current_step = 0
        self.current_weight = start_weight

    def step(self):
        self.current_step += 1
        progress = self.current_step / self.total_steps
        self.current_weight = self.start_weight + (self.max_weight - self.start_weight) * progress
        # Clip the weight to be within the specified range
        self.current_weight = min(max(self.current_weight, self.start_weight), self.max_weight)
        return self.current_weight

    def get_weight(self):
        return self.current_weight
    
class LinearDiscriminator(torch.nn.Module):
    def __init__(self, input_features):
        super(LinearDiscriminator, self).__init__()
        self.linear = torch.nn.Linear(input_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class ConditionalBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.embed = torch.nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        # y is the condition, i.e. batch number
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.expand_as(out)
        beta = beta.expand_as(out)
        return gamma * out + beta
    
class ConditionalBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm2d(num_features, affine=False)
        self.embed = torch.nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        # y is the condition, i.e. batch number
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(out)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(out)
        return gamma * out + beta
    
# def add_encoder_batch_norm(model, 
#                            BatchNorm, 
#                            n_conditions=None
#                            ):
#     new_layers = []
#     for layer in model:
#         new_layers.append(layer)
#         if isinstance(layer, nn.Conv2d):
#             if BatchNorm == ConditionalBatchNorm2d:
#                 new_layers.append(BatchNorm(layer.out_channels, n_conditions))
#             else:
#                 new_layers.append(BatchNorm(layer.out_channels))
#     return nn.Sequential(*new_layers)

# def add_decoder_batch_norm(linear, 
#                            net, 
#                            BatchNorm1d=nn.BatchNorm1d, 
#                            BatchNorm2d=nn.BatchNorm2d, 
#                            n_conditions=None):
#     new_linear = []
#     for layer in linear:
#         new_linear.append(layer)
#         if isinstance(layer, nn.Linear):
#             if BatchNorm1d == ConditionalBatchNorm1d:
#                 new_linear.append(BatchNorm1d(layer.out_features, n_conditions))
#             else:
#                 new_linear.append(BatchNorm1d(layer.out_features))
    
#     new_net = []
#     for i, layer in enumerate(net):
#         new_net.append(layer)
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)) and i != len(net):
#             if BatchNorm2d == ConditionalBatchNorm2d:
#                 new_net.append(BatchNorm2d(layer.out_channels, n_conditions))
#             else:
#                 new_net.append(BatchNorm2d(layer.out_channels))

#     return nn.Sequential(*new_linear), nn.Sequential(*new_net)
