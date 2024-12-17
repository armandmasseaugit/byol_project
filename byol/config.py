import torch.nn as nn
from torchvision import transforms

#################################################################################
# Global Parameters and Configuration                                            #
#################################################################################

# Data Transformation Parameters
# These transformations are applied to each view of the dataset
TRANSFORMS = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
    ]
)

BATCH_SIZE = 64  # Batch size for training (adjustable)
SHUFFLE = True  # Whether to shuffle the dataset

#################################################################################
# Model Architecture Parameters                                                  #
#################################################################################

# Moving Average Parameter for Momentum (used in BYOL)
TAU = 0.9

# Encoder network (converts image to representation)
ENCODER = nn.Sequential(
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

# Projection head (projects the encoder's output to a smaller space)
PROJECTION_DIM = 32  # The dimension of the projection space (typically >128)
PROJECTOR = nn.Sequential(
    nn.Linear(64, PROJECTION_DIM),
    nn.ReLU(),
)

# Predictor network (used to predict the projected representations of the target network)
PREDICTOR = nn.Sequential(
    nn.Linear(PROJECTION_DIM, 32),
    nn.ReLU(),
    nn.Linear(32, PROJECTION_DIM),
)

# Fine-tuning MLP (for supervised fine-tuning after pre-training)
FINE_TUNING_MLP = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),  # 10 output units for classification (MNIST)
    # nn.Softmax(dim=-1)
)


#################################################################################
# Training Parameters                                                            #
#################################################################################

NUM_EPOCHS = 10  # Number of training epochs
PATH_OF_THE_SAVED_MODEL_PARAMETERS = (
    "models/trained_byol_model.pth"  # Path to save the pre-trained encoder's parameters
)

#################################################################################
# Testing Parameters                                                             #
#################################################################################

PATH_OF_THE_MODEL_TO_TEST = (
    "models/fine-tuned_model.pth"  # Path to load the fine-tuned model for evaluation
)
