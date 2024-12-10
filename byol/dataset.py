from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from .config import TRANSFORMS, BATCH_SIZE, SHUFFLE

class MnistDataset:
    def __init__(self):
        """
            MNIST dataset ready for training.

            Attributes
            ----------

            Methods
            -------
            get_train_loader
                get the train DataLoader.
            get_test_loader
                get the test DataLoader.
        """
        self.train_data = MNIST(root="data/raw", train=True, download=True, transform=TRANSFORMS)
        self.test_data = MNIST(root="data/raw", train=False, download=True, transform=TRANSFORMS)

        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE
        )
        self.test_loader = DataLoader(
            dataset=self.test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE
        )

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


if __name__ == "__main__":
    m = MnistDataset()
    print(m.get_train_loader())
