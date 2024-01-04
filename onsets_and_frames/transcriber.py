"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch as t
import torch.nn.functional as F
from torch import nn

from nnAudio.features import CQT

from .network_utils import From3Dto2D, From2Dto3D, FGLSTM
from .harmonic_layers import HarmonicDilatedConv, HarmConvBlock
from .constants import HOP_LENGTH

from typing import Tuple


class HPPNet(nn.Module):
    def __init__(self, bins_per_octave: int = 48, desired_frame_rate: int = 50, sr: int = 16000, n_dilated_conv_layers: int = 3,
                 convblock_length: int = 3, add_dilated_convblock: bool = True, post_dilated_convblock_length: int = 3, channel_sizes: Tuple[int, int, int] = [16, 128, 128]):
        super(HPPNet, self).__init__()
        self.sr = sr
        self.num_octaves = 8
        self.bins_per_octave = bins_per_octave
        self.n_bins_in = self.bins_per_octave * self.num_octaves
        self.n_dilated_conv_layers = n_dilated_conv_layers
        self.convblock_length = convblock_length
        self.add_dilated_convblock = add_dilated_convblock
        self.post_dilated_convblock_length = post_dilated_convblock_length
        self.channel_sizes = channel_sizes

        # CQT extractor
        self.hopsize = HOP_LENGTH
        self.feature_rate = int(self.sr/self.hopsize)
        self.cqt_layer = CQT(bins_per_octave=self.bins_per_octave, n_bins=352, hop_length=self.hopsize, sr=sr, 
                             pad_mode="constant", center=False, trainable=False, verbose=False)
        
        convblock_0 = nn.Sequential(
            nn.Conv2d(kernel_size=(7, 7), in_channels=1, out_channels=self.channel_sizes[0], groups=1, padding="same"),
            nn.ReLU(),
            nn.InstanceNorm2d(self.channel_sizes[0]),
        )

        convblock = nn.Sequential(
            nn.Conv2d(kernel_size=(7, 7), in_channels=self.channel_sizes[0], out_channels=self.channel_sizes[0], groups=1, padding="same"),
            nn.ReLU(),
            nn.InstanceNorm2d(self.channel_sizes[0]),
        )

        self.convblock = nn.Sequential()
        for i, _ in enumerate(range(self.convblock_length)):
            if i == 0:
                self.convblock = self.convblock.append(convblock_0)
            else:
                self.convblock = self.convblock.append(convblock)

        self.harmonic_block = HarmonicDilatedConv(c_in=self.channel_sizes[0], c_out=self.channel_sizes[1])

        dilated_convblock_base = nn.Sequential(
                nn.Conv2d(kernel_size=(3, 1), in_channels=self.channel_sizes[1], out_channels=self.channel_sizes[2], groups=1, padding="same",
                    dilation=(48, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(4, 1)),
                nn.InstanceNorm2d(self.channel_sizes[2]),

                nn.Conv2d(kernel_size=(3, 1), in_channels=self.channel_sizes[2], out_channels=self.channel_sizes[2], groups=1, padding="same",
                    dilation=(12, 1)),
                nn.ReLU(),
                nn.InstanceNorm2d(self.channel_sizes[2]),
        )
        
        dilated_convblock_extension = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 5), in_channels=self.channel_sizes[2], out_channels=self.channel_sizes[2], groups=1, padding="same"),
            nn.ReLU(),
            nn.InstanceNorm2d(self.channel_sizes[2]),
        )

        dilated_convblock = dilated_convblock_base if self.add_dilated_convblock else nn.Sequential(nn.MaxPool2d(kernel_size=(4, 1)))

        for _ in range(self.post_dilated_convblock_length):
            dilated_convblock = dilated_convblock.append(dilated_convblock_extension)

        self.dilated_convblock = dilated_convblock

        self.fglstm_frames = FGLSTM(channel_in=self.channel_sizes[2], channel_out=1, lstm_size=self.channel_sizes[2])
        self.fglstm_onsets = FGLSTM(channel_in=self.channel_sizes[2], channel_out=1, lstm_size=self.channel_sizes[2])
        self.fglstm_offsets = FGLSTM(channel_in=self.channel_sizes[2], channel_out=1, lstm_size=self.channel_sizes[2])
        self.fglstm_velocities = FGLSTM(channel_in=self.channel_sizes[2], channel_out=1, lstm_size=self.channel_sizes[2])

    def obtain_cqt(self, x):
        kernel_size = self.cqt_layer.cqt_kernels_imag.shape[-1]
        x = F.pad(input=x, pad=(kernel_size//2, kernel_size//2), mode="constant", value=0)
        cqt = self.cqt_layer(x)
        return t.log10(cqt + 1)

    def neural_processing(self, cqt):
        output = cqt.unsqueeze(1)
        output = self.convblock(output)
        output = self.harmonic_block(output)
        output = self.dilated_convblock(output)
        frames = self.fglstm_frames(output)
        onsets = self.fglstm_onsets(output)
        offsets = self.fglstm_offsets(output)
        velocities = self.fglstm_velocities(output)
        return frames, onsets, offsets, velocities

    def forward(self, x):
        cqt = self.obtain_cqt(x)
        frames, onsets, offsets, velocities = self.neural_processing(cqt)

        frames = t.sigmoid(frames)
        onsets = t.sigmoid(onsets)
        offsets = t.sigmoid(offsets)
        velocities = t.sigmoid(velocities)
        return frames, onsets, offsets, velocities

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        frames = onset_label.size()[-2]
        frame_pred, onset_pred, offset_pred, velocity_pred = self.forward(audio_label)
        frame_pred, onset_pred, offset_pred, velocity_pred = frame_pred[..., :frames, :], onset_pred[..., :frames, :], offset_pred[..., :frames, :], velocity_pred[..., :frames, :]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': (-2 * onset_label * t.log(predictions["onset"]) - (1 - onset_label) * t.log(1-predictions["onset"])).mean(),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator



class HPPNetLess(HPPNet):
    """
    HPPNet without harmonic knowledge
    """
    def __init__(self, **kwargs):
        super(HPPNetLess, self).__init__(**kwargs)
        self.harmonic_block = nn.Conv2d(kernel_size=(3, 1), in_channels=16, out_channels=128, groups=1, dilation=(1, 1), padding="same")

############ 3D Models

class HPPNetDDD(HPPNet):
    def __init__(self, **kwargs):
        super(HPPNetDDD, self).__init__(**kwargs)

        self.cqt_layer = CQT(bins_per_octave=self.bins_per_octave, fmax=self.sr//2, hop_length=self.hopsize, sr=self.sr,
                             pad_mode="constant", center=False, trainable=False, verbose=False)

        self.harmonic_block = nn.Sequential(
            From2Dto3D(bins_per_octave=self.bins_per_octave, n_octaves=self.num_octaves),
            HarmConvBlock(n_in_channels=self.channel_sizes[0], n_out_channels=self.channel_sizes[1], octave_depth=3, dilation_rates=[4*0, 4*7, 4*4, 4*10]),
            From3Dto2D(bins_per_octave=self.bins_per_octave, n_octaves=self.num_octaves),
        )

    def neural_processing(self, cqt):
        output = cqt.unsqueeze(1)
        output = self.convblock(output)
        output = self.harmonic_block(output)
        output = self.dilated_convblock(output)
        output = output[:, :, 4:-4, :]  # Output 3rd is dimension 96, should be 88 for piano
        frames = self.fglstm_frames(output)
        onsets = self.fglstm_onsets(output)
        offsets = self.fglstm_offsets(output)
        velocities = self.fglstm_velocities(output)
        return frames, onsets, offsets, velocities
