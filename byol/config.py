from torchvision import transforms
import torch.nn as nn

# Here are stored the parameters

### Data transformation parameters

# The transformations of the dataset in order to create two views
transforms = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet parameters
    ]
)

BATCH_SIZE = 64  # 512

SHUFFLE = True

### Model parameters

# Parameter of the moving average
TAU = 0.9

# Optimizer
LEARNING_RATE = 0.0003

# Encoder (from view to representation)
encoder = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),  # (batch_size, 6, 24, 24)
    nn.ReLU(),
    nn.MaxPool2d(2),  # (batch_size, 6, 12, 12)
    nn.Conv2d(6, 16, kernel_size=5),  # (batch_size, 16, 8, 8)
    nn.ReLU(),
    nn.MaxPool2d(2),  # (batch_size, 16, 4, 4)
    nn.Flatten(),  # (batch_size, 16*4*4)
    nn.Linear(16 * 4 * 4, 120),
    nn.ReLU(),
    nn.Linear(120, 64),
)

PROJECTION_DIM = 32  # >128 for the moment

# Projector
projector = nn.Sequential(
    nn.Linear(64, PROJECTION_DIM),
    nn.ReLU(),
)
# Predictor
predictor = nn.Sequential(
    nn.Linear(PROJECTION_DIM, 32),
    nn.ReLU(),
    nn.Linear(32, PROJECTION_DIM),
)

# Fine-tuning mlp
fine_tuning_mlp = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    # nn.Softmax(dim=-1)
)


# Loss function
def loss_function(x, y):
    x = nn.functional.normalize(x, dim=-1)
    y = nn.functional.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


### Training parameters

NUM_EPOCHS = 10
PATH_OF_THE_SAVED_MODEL_PARAMETERS = "models/trained_byol_model.pth"  # Encoder's parameters

### Testing parameters

PATH_OF_THE_MODEL_TO_TEST = "models/fine-tuned_model.pth"
