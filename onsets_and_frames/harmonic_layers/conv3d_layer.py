import torch as t
import torch.nn as nn
import torch.nn.functional as F

from ..network_utils import CircularOctavePadding, Conv2D, Conv3D

from typing import Optional, Iterable


class HarmConvBlock(nn.Module):
    """
    For our 3D harmonic convolutions
    """
    def __init__(self, n_in_channels: int, n_out_channels: int, octave_depth: int = 3,
                 dilation_rates: Optional[Iterable[int]] = None, time_width: int = 1, special_padding: bool = True, depthwise: bool = False,
                 ):
        super(HarmConvBlock, self).__init__()
        if dilation_rates is None:
            dilation_rates = [0, 28, 16]
        self.dilation_rates = dilation_rates
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.octave_depth = octave_depth
        self.time_width = time_width
        self.special_padding = special_padding
        self.use_always_3d = False
        self.using_3d_convolutions = False
        self.depthwise = depthwise

        if self.special_padding:
            padding = 0
            padding_layer = CircularOctavePadding
        else:
            padding = "same"
            padding_layer = nn.Identity
        
        if self.time_width > 1 or self.use_always_3d:
            self.using_3d_convolutions = True

        module_list = []
        for dl in dilation_rates:
            if dl == 0:
                kernel_size_h = 1
                dilation = 1
            else:
                kernel_size_h = 2
                dilation = dl

            if self.using_3d_convolutions:
                module_list.append(
                    nn.Sequential(
                        padding_layer(kernel_size=(octave_depth, kernel_size_h, time_width), pitch_class_dilation=dilation),
                        Conv3D(n_in_channels, n_out_channels, padding_mode="circular", kernel_size=(octave_depth, kernel_size_h, time_width), padding=padding, dilation=(1, dilation, 1), depthwise=depthwise)
                    )
                )
            else:
                module_list.append(
                    nn.Sequential(
                        padding_layer(kernel_size=(octave_depth, kernel_size_h, time_width), pitch_class_dilation=dilation),
                        Conv2D(n_in_channels, n_out_channels, padding_mode="circular", kernel_size=(octave_depth, kernel_size_h), padding=padding, dilation=(1, dilation), depthwise=depthwise)
                    )
                )

        self.module_list = nn.ModuleList(module_list)

    def forward_2d(self, x):
        batch, channels, octaves, pitch_classes, frames = x.size()
        x = x.permute([0, 4, 1, 2, 3]).reshape([batch * frames, channels, octaves, pitch_classes])  # Stack frames at batch dimension
        outputs = None
        for module in self.module_list:
            if outputs is None:
                outputs = module(x)
            else:
                outputs += module(x)
        # Return to the original shape
        outputs = outputs.reshape([batch, frames, self.n_out_channels, octaves, pitch_classes]).permute([0, 2, 3, 4, 1])
        return outputs
    
    def forward_3d(self, x):
        outputs = None
        for module in self.module_list:
            if outputs is None:
                outputs = module(x)
            else:
                outputs += module(x)
        return outputs

    def forward(self, x):
        if self.using_3d_convolutions:
            output = self.forward_3d(x)
        else:
            output = self.forward_2d(x)
        return F.relu(output)