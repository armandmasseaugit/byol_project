from torch.utils.data import DataLoader, Dataset


class BYOLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        view1 = self.transform(image)  # We generate two views for the same image
        view2 = self.transform(image)
        return view1, view2, label
