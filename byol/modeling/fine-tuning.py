from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import device, cuda, save, max, load
import torch.nn as nn
from torchvision.transforms.v2 import ToTensor

from byol.dataset import BYOLDataset
from byol.modeling.model import FineTunedBootstrapYourOwnLatent, Encoder
from byol.config import (encoder, NUM_EPOCHS, BATCH_SIZE, SHUFFLE, TAU,
                         PATH_OF_THE_SAVED_MODEL_PARAMETERS, fine_tuning_mlp)
import torch.optim as optim

encoder = Encoder(encoder)
encoder.load_state_dict(load(PATH_OF_THE_SAVED_MODEL_PARAMETERS), strict=False)
for param in encoder.parameters():
    param.requires_grad = False

model = FineTunedBootstrapYourOwnLatent(encoder, fine_tuning_mlp)

device = device("cuda" if cuda.is_available() else "cpu")
model = model.to(device)

dataset = MNIST(root="data/raw", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

optimizer = optim.Adam(model.mlp.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


state_dict = model.state_dict()
state_dict_renamed = {k.replace("encoder.encoder", "encoder"): v for k, v in state_dict.items()}

save(state_dict_renamed, "models/fine-tuned_model.pth")

