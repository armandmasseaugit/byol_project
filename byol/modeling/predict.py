from byol.modeling.model import FineTunedBootstrapYourOwnLatent, Encoder
from torch import load, no_grad, device, cuda, max
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from byol.config import (
    ENCODER,
    BATCH_SIZE,
    SHUFFLE,
    PATH_OF_THE_MODEL_TO_TEST,
    FINE_TUNING_MLP,
)

encoder = Encoder(ENCODER)
model = FineTunedBootstrapYourOwnLatent(encoder, FINE_TUNING_MLP)

model.load_state_dict(load(PATH_OF_THE_MODEL_TO_TEST))

device = device("cuda" if cuda.is_available() else "cpu")
model.to(device)

model.eval()
dataset = MNIST(root="data/raw", train=False, download=True, transform=ToTensor())

test_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

total = 0
correct = 0
with no_grad():
    for index, (view, labels) in enumerate(test_dataloader):
        view = view.to(device)
        labels = labels.to(device)

        output = model(view)

        _, predicted = max(output, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        if index % 10 == 0:
            print(f"Batch [{index}/{len(test_dataloader)}], Accuracy: {correct / total:.4f}")

accuracy = 100 * correct / total
print(f"Test completed. Accuracy: {accuracy:.2f}%")
