from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import device, cuda, save
from byol.dataset import BYOLDataset
from byol.modeling.model import BootstrapYourOwnLatent, Encoder, Projector, Predictor
from byol.config import (
    ENCODER,
    PROJECTOR,
    PREDICTOR,
    NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING,
    TRANSFORMS,
    BATCH_SIZE,
    SHUFFLE,
    TAU,
    PATH_OF_THE_SAVED_MODEL_PARAMETERS,
)
import torch.optim as optim

encoder = Encoder(ENCODER)
projector = Projector(PROJECTOR)
predictor = Predictor(PREDICTOR)

model = BootstrapYourOwnLatent(encoder, projector, predictor, TAU)
device = device("cuda" if cuda.is_available() else "cpu")
model = model.to(device)

dataset = MNIST(root="data/raw", train=True, download=True)
transformed_dataset = BYOLDataset(dataset, TRANSFORMS)
# TODO: add the downloading of processed data at data/processed
train_dataloader = DataLoader(dataset=transformed_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

model.train()
for epoch in range(NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING):
    total_loss = 0
    for index, (view1, view2, label) in enumerate(train_dataloader):

        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        loss = model(view1, view2)

        loss.backward()
        optimizer.step()
        model.update_the_moving_average_for_the_encoder()
        model.update_the_moving_average_for_the_projector()

        total_loss += loss.item()
        if index % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING}], Iter [{index + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
            )
    print(f"Epoch {epoch+1} completed. Average loss: {total_loss/len(train_dataloader):.4f}")
print("Training ended !")

save(model.online_encoder.state_dict(), PATH_OF_THE_SAVED_MODEL_PARAMETERS)
