from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import device, cuda, save, max, load, equal
import torch.nn as nn
from torchvision.transforms.v2 import ToTensor

from byol.modeling.model import FineTunedBootstrapYourOwnLatent, Encoder
from byol.config import (
    ENCODER,
    NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING,
    BATCH_SIZE,
    SHUFFLE,
    TAU,
    PATH_OF_THE_SAVED_MODEL_PARAMETERS,
    FINE_TUNING_MLP,
    PATH_OF_THE_SAVED_FINE_TUNING_PARAMETERS,
)
import torch.optim as optim

encoder = Encoder(ENCODER)

encoder.load_state_dict(load(PATH_OF_THE_SAVED_MODEL_PARAMETERS, weights_only=True))

model = FineTunedBootstrapYourOwnLatent(encoder, FINE_TUNING_MLP)

device = device("cuda" if cuda.is_available() else "cpu")
model = model.to(device)

dataset = MNIST(root="data/raw", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

optimizer = optim.SGD(model.mlp.parameters(), lr=1e-3, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# encoder_parameters_before_training = list(model.encoder.parameters())

for epoch in range(NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING):
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

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

encoder_parameters_after_training = list(model.encoder.parameters())

# print(all(equal(p1, p2) for p1, p2 in zip(encoder_parameters_before_training, encoder_parameters_after_training)))

state_dict = model.state_dict()

save(state_dict, PATH_OF_THE_SAVED_FINE_TUNING_PARAMETERS)