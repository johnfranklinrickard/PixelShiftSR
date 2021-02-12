import numpy
from os import listdir
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


class Loader:
    def __init__(self, mode):
        train_set = ShiftXYDataLoader("train-data", mode)
        val_set = ShiftXYDataLoader("val-data", mode)

        self.train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)


# This class populates the dataset with relevant data for training/testing
class ShiftXYDataLoader(Dataset):
    def __init__(self, folder_name, data_folder):
        super(ShiftXYDataLoader, self).__init__()

        # data paths are relative but hard coded, especially the folder naming structure
        self.image_dir = f"./{folder_name}/{data_folder}/{data_folder}-"
        self.target_dir = f"./{folder_name}/original/original-"
        self.number_of_images = self.count_images(f"./{folder_name}/{data_folder}")

        # load additional x and y positional data
        self.x_coords, self.y_coords = self.load_xy(folder_name, data_folder)
        self.original_x, self.original_y = self.load_xy(folder_name, "original")

    @staticmethod
    def count_images(directory):
        return len([f for f in listdir(directory) if f.endswith('.png')])

    @staticmethod
    def load_xy(top_folder, data_folder):
        x = numpy.loadtxt(f"./{top_folder}/{data_folder}/x_coords.txt")
        y = numpy.loadtxt(f"./{top_folder}/{data_folder}/y_coords.txt")
        image_size = 64.0
        # normalize values in arrays between -1 and 1
        x = x / (image_size / 2.0) - 1.0
        y = y / (image_size / 2.0) - 1.0
        return x.astype(numpy.float32), y.astype(numpy.float32)

    def __getitem__(self, index):
        image, target = self.open_images(index)
        image = image.numpy()[0, :, :]
        target = target.numpy()[0, :, :]
        
        # stacks the input data with the corresponding x and y data
        first = numpy.stack([image, self.x_coords, self.y_coords], axis=2)

        # this is the one channel output
        second = numpy.stack([target], axis=2)

        # this is the output with x and y added to the targets if necessary
        # second = np.stack([target, self.original_x, self.original_y], axis=2)
        return ToTensor()(first), ToTensor()(second)

    def __len__(self):
        return self.number_of_images

    def open_images(self, index):
        image, _, _ = Image.open(f"{self.image_dir}{index:05d}.png").convert('YCbCr').split()
        target, _, _ = Image.open(f"{self.target_dir}{index:05d}.png").convert('YCbCr').split()
        return ToTensor()(image), ToTensor()(target)
