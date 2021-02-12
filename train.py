import torch
import torchnet
import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from loader import Loader
from logger import Logger
from model import NetXY
from psnrmeter import PSNRXYMeter


def processor(sample):
    data, target, training = sample
    data = torch.autograd.Variable(data)
    target = torch.autograd.Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)

    return loss, output


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    scheduler.step()
    state['iterator'] = tqdm.tqdm(state['iterator'])


def on_end_epoch(state):
    print('[%s][Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
        current_mode, state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    logger.log_train_loss(state['epoch'], meter_loss.value()[0])
    logger.log_train_psnr(state['epoch'], meter_psnr.value())

    reset_meters()

    engine.test(processor, loader.val_loader)
    logger.log_val_loss(state['epoch'], meter_loss.value()[0])
    logger.log_val_psnr(state['epoch'], meter_psnr.value())

    print('[%s][Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        current_mode, state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), f"epochs/epoch_{UPSCALE_FACTOR}_{current_mode}_XY_{state['epoch']}.pt")


def setup_model_criterion(upscale_factor):
    _model = NetXY(upscale_factor)
    _criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        _model = _model.cuda()
        _criterion = _criterion.cuda()

    return _model, _criterion


def setup_engine_with_hooks():
    _engine = torchnet.engine.Engine()
    _engine.hooks['on_sample'] = on_sample
    _engine.hooks['on_forward'] = on_forward
    _engine.hooks['on_start_epoch'] = on_start_epoch
    _engine.hooks['on_end_epoch'] = on_end_epoch
    return _engine


if __name__ == "__main__":

    # variables are mostly global for ease of access as this is a proof of concept
    UPSCALE_FACTOR = 4  # constant
    num_epochs = 2  # change this to adjust training period

    print(f"{torch.cuda.is_available()}")

    # train different modes
    for mode in ["control", "half", "quarter", "random0"]:
        current_mode = mode
        logger = Logger(mode, "oneOutputChannel")
        loader = Loader(mode)

        model, criterion = setup_model_criterion(UPSCALE_FACTOR)
        print('# parameters:', sum(param.numel() for param in model.parameters()))

        meter_loss = torchnet.meter.AverageValueMeter()
        meter_psnr = PSNRXYMeter()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

        engine = setup_engine_with_hooks()
        engine.train(processor, loader.train_loader, maxepoch=num_epochs, optimizer=optimizer)

        logger.save_state()
