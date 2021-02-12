from torch.nn import Module, Conv2d, PixelShuffle
from torch.nn.functional import tanh, sigmoid


class NetXY(Module):
    def __init__(self, upscale_factor):
        super(NetXY, self).__init__()

        self.conv1 = Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # Here the 1 multiplicator can be 3 if the x and y outputs are wanted
        self.conv3 = Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = PixelShuffle(upscale_factor)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = tanh(self.conv2(x))
        x = sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
