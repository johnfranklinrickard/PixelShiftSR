from torchnet.logger import VisdomPlotLogger, VisdomSaver


class Logger:
    def __init__(self, mode, name):
        self.name = name
        self.train_loss = self.__create_logger(self.name, f"{mode} Train Loss")
        self.train_psnr = self.__create_logger(self.name, f"{mode} Train PSNR")
        self.val_loss = self.__create_logger(self.name, f"{mode} Val Loss")
        self.val_psnr = self.__create_logger(self.name, f"{mode} Val PSNR")

    @staticmethod
    def __create_logger(name, title):
        return VisdomPlotLogger('line', env=name, opts={'title': title})

    def log_train_loss(self, state, value):
        self.train_loss.log(state, value)
        return

    def log_train_psnr(self, state, value):
        self.train_psnr.log(state, value)
        return

    def log_val_loss(self, state, value):
        self.val_loss.log(state, value)
        return

    def log_val_psnr(self, state, value):
        self.val_psnr.log(state, value)
        return

    def save_state(self):
        VisdomSaver(envs=[self.name]).save()
        return
