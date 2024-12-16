from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import device, cuda, save
from byol.dataset import BYOLDataset
from byol.modeling.model import BootstrapYourOwnLatent, Encoder, Projector, Predictor
from byol.config import (
    encoder,
    projector,
    predictor,
    loss_function,
    NUM_EPOCHS,
    transforms,
    BATCH_SIZE,
    SHUFFLE,
    TAU,
    PATH_OF_THE_SAVED_MODEL_PARAMETERS,
)
import torch.optim as optim

encoder_ = Encoder(encoder)
projector_ = Projector(projector)
predictor_ = Predictor(predictor)

model = BootstrapYourOwnLatent(encoder_, projector_, predictor_, loss_function, TAU)
device = device("cuda" if cuda.is_available() else "cpu")
model = model.to(device)

dataset = MNIST(root="data/raw", train=True, download=True)
transformed_dataset = BYOLDataset(dataset, transforms)
# TODO: add the downloading of processed data at data/processed
train_dataloader = DataLoader(dataset=transformed_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

print(len(train_dataloader))

optimizer = optim.Adam(model.parameters(), lr=0.0003)

total_loss = 0
model.train()
for epoch in range(NUM_EPOCHS):
    for index, (view1, view2, label) in enumerate(train_dataloader):

        # image1, image2 = image1.cuda(), image2.cuda()
        optimizer.zero_grad()
        loss = model(view1, view2)

        loss.backward()
        optimizer.step()
        model.update_the_moving_average_for_the_encoder()
        model.update_the_moving_average_for_the_projector()

        total_loss += loss.item()
        if index % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Iter [{index + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
            )
    print(f"Epoch {epoch+1} completed. Average loss: {total_loss/len(train_dataloader):.4f}")
print("Training ended !")

save(model.online_encoder.state_dict(), PATH_OF_THE_SAVED_MODEL_PARAMETERS)
