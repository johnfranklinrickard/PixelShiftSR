from math import log10
from torchnet.meter import meter
import torch


class PSNRXYMeter(meter.Meter):
    def __init__(self):
        super(PSNRXYMeter, self).__init__()
        self.count = 0
        self.total = 0.0

    def reset(self):
        self.count = 0
        self.total = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        output = output.cpu()
        target = target.cpu()

        output = output[:, 0, :, :]
        target = target[:, 0, :, :]
        self.count += output.numel()
        self.total += torch.sum((output - target) ** 2)

    def value(self):
        mse = self.total / max(1, self.count)
        psnr = 10 * log10(1 / mse)
        return psnr
