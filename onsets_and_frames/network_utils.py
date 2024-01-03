import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple


def compute_hopsize_cqt(fs_cqt_target, sr=16000):
    """
    Computes the necessary CQT hopsize to approximate a desired feature rate fs_cqt_target
    Args:
        fs_cqt_target:  desired frame rate in Hz
        fs:             audio sampling rate
        num_octaves:    number of octaves for the CQT
    Returns:
        hopsize_cqt:    CQT hopsize in samples
        fs_cqt:         resulting CQT frame rate in Hz
    """
    hopsize_target = sr // fs_cqt_target
    return hopsize_target


class CircularOctavePadding(nn.Module):
    def __init__(
            self, kernel_size: Tuple[int], pitch_class_dilation: int, strides=Optional[None]
    ) -> None:
        super(CircularOctavePadding, self).__init__()
        self.kernel_size = kernel_size
        self.pitch_class_dilation = pitch_class_dilation
        self.strides = strides  # Not implemented
        self.dummy_padding = nn.ConstantPad1d(1, 0)
        self.pitch_class_required_padding = (
            0 if kernel_size[1] == 1 else self.pitch_class_dilation
        )

    def forward(self, x):
        try:  # Full 3D convolution
            batch, channels, octaves, pitch_classes, frames = x.size()
            pitch_class_padding = x[:, :, :, :self.pitch_class_required_padding, :].roll(-1, dims=2)
            pitch_class_padding[:, :, -1, :, :] = 0
            octave_padding = t.zeros(
                (batch, channels, self.kernel_size[0]-1, pitch_classes + self.pitch_class_required_padding, frames),
                device=x.device
            )

            if self.pitch_class_required_padding > 0:
                padded_x = t.concat([x, pitch_class_padding], dim=-2)
                padded_x = t.concat([padded_x, octave_padding], dim=-3)
            else:
                padded_x = t.concat([x, octave_padding], dim=-3)
        except:  # 2D trick
            batch, channels, octaves, pitch_classes = x.size()
            pitch_class_padding = x[:, :, :, :self.pitch_class_required_padding].roll(-1, dims=2)
            pitch_class_padding[:, :, -1, :] = 0
            octave_padding = t.zeros(
                (batch, channels, self.kernel_size[0]-1, pitch_classes + self.pitch_class_required_padding),
                device=x.device
            )

            if self.pitch_class_required_padding > 0:
                padded_x = t.concat([x, pitch_class_padding], dim=-1)
                padded_x = t.concat([padded_x, octave_padding], dim=-2)
            else:
                padded_x = t.concat([x, octave_padding], dim=-2)
        return padded_x


class MultiRateConv(nn.Module):
    """
    For HarmoF0
    """
    def __init__(self, n_in_channels, n_out_channels, dilation_rates: Optional[Iterable[int]]):
        super(MultiRateConv, self).__init__()
        if dilation_rates is None:
            dilations_rates = [0, 28, 76]
        self.dilation_rates = dilation_rates
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels

        module_list = []
        for _ in dilations_rates:
            module_list.append(
                nn.Conv2d(n_in_channels, n_out_channels, padding_mode="circular", kernel_size=(3, 1), padding="same", dilation=(1, 1))
            )
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        outputs = [module(x) for module in self.module_list]
        for idx, shift in enumerate(self.dilation_rates):
            outputs[idx] = t.roll(outputs[idx], -shift, dims=1)
            if shift > 0:
                outputs[idx][:, :, -shift:, :] = 0
        return t.stack(outputs, dim=1).sum(dim=1)


class From2Dto3D(nn.Module):
    def __init__(self, bins_per_octave: int, n_octaves: int):
        super(From2Dto3D, self).__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = int(self.n_octaves * self.bins_per_octave)

    def forward(self, cqt):
        padding_needed = self.total_bins - cqt.shape[2]
        cqt = F.pad(cqt, (0, 0, 0, padding_needed))
        batch, channels, bins, frames = cqt.size()
        octave_pc_spectrum = cqt.reshape([batch, channels, self.n_octaves, self.bins_per_octave, frames])
        return octave_pc_spectrum


class From3Dto2D(nn.Module):
    def __init__(self, bins_per_octave: int, n_octaves: int):
        super(From3Dto2D, self).__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = int(self.n_octaves * self.bins_per_octave)

    def forward(self, octave_pc_spectrum):
        batch, channels, octaves, pitch_classes, frames = octave_pc_spectrum.size()
        cqt = octave_pc_spectrum.reshape([batch, channels, self.total_bins, frames])
        return cqt


class FGLSTM(nn.Module):
    """
    From hhpnet code
    """

    def __init__(self, channel_in, channel_out, lstm_size) -> None:
        super().__init__()

        self.channel_out = channel_out

        self.lstm = BiLSTM(channel_in, lstm_size // 2)
        self.linear = nn.Linear(lstm_size, channel_out)

    def forward(self, x):
        # inputs: [b x c_in x freq x T]
        # outputs: [b x c_out x T x freq]

        b, c_in, n_freq, frames = x.size()

        # => [b x freq x T x c_in]
        x = t.permute(x, [0, 3, 2, 1])

        # => [(b*freq) x T x c_in]
        x = x.reshape([b * n_freq, frames, c_in])
        # => [(b*freq) x T x lstm_size]
        x = self.lstm(x)
        # => [(b*freq) x T x c_out]
        x = self.linear(x)
        # => [b x freq x T x c_out]
        x = x.reshape([b, n_freq, frames, self.channel_out])
        # => [b x c_out x T x freq]
        x = t.permute(x, [0, 3, 2, 1])
        return x.squeeze(1)


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = t.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = t.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = t.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                # h.zero_()
                # c.zero_()
                # ONNX does not support tensor.zero_(), so use following:
                h.fill_(0)
                c.fill_(0)

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depthwise: bool = False, **kwargs) -> None:
        super(Conv2D, self).__init__()
        self.depthwise = depthwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_in_ratio = None
        self.groups = 1

        if depthwise:
            self.n_groups = in_channels
            self.out_in_ratio = in_channels/out_channels
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=self.groups, **kwargs),
                nn.ReLU(),
                nn.InstanceNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, groups=self.groups, kernel_size=(1, 1))
            )
        else:
            self.block = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, groups=self.groups, **kwargs)
    
    def forward(self, x):
        return self.block(x)
    

class Conv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depthwise: bool = False, **kwargs) -> None:
        super(Conv2D, self).__init__()
        self.depthwise = depthwise
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_in_ratio = None
        self.groups = 1

        if depthwise:
            self.n_groups = in_channels
            self.out_in_ratio = in_channels/out_channels
            self.block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=in_channels, groups=self.groups, **kwargs),
                nn.ReLU(),
                nn.InstanceNorm2d(in_channels),
                nn.Conv3d(in_channels, out_channels, groups=self.groups, kernel_size=(1, 1, 1))
            )
        else:
            self.block = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, groups=self.groups, **kwargs)
    
    def forward(self, x):
        return self.block(x)