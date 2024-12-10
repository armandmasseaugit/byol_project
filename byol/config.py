from torchvision import transforms
# Here are stored the parameters

### Data transformation parameters

# The transformations of the dataset in order to create two views
TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet parameters
])

BATCH_SIZE=32

SHUFFLE=True

### Model parameters
ACTIVATION_FUNCTION = 'relu' # relu, sigmoïd
LAST_ACTIVATION_FUNCTION = 'softmax' # TODO: add other functions

### Training parameters
