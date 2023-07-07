import torch.nn as nn
from ..network_utils import Conv2D
import torch.nn.functional as F

class HarmonicDilatedConv(nn.Module):
    """
    From the HPPNet original code. It is fixed to 4bins per semitone (see dilation)
    """

    def __init__(self, c_in, c_out, depthwise: bool = False) -> None:
        super(HarmonicDilatedConv, self).__init__()
        super().__init__()
        self.conv_1 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[48, 1])
        self.conv_2 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[76, 1])
        self.conv_3 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[96, 1])
        self.conv_4 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[111, 1])
        self.conv_5 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[124, 1])
        self.conv_6 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[135, 1])
        self.conv_7 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[144, 1])
        self.conv_8 = Conv2D(c_in, c_out, depthwise = depthwise, kernel_size = [3, 1], padding="same", dilation=[152, 1])

    def forward(self, x):
        x = (
            self.conv_1(x)
            + self.conv_2(x)
            + self.conv_3(x)
            + self.conv_4(x)
            + self.conv_5(x)
            + self.conv_6(x)
            + self.conv_7(x)
            + self.conv_8(x)
        )
        return F.relu(x)