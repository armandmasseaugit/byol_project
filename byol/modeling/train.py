from byol.dataset import MnistDataset

dataset_ = MnistDataset()
train = dataset_.get_train_loader()
