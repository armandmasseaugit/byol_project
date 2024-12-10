import torch
import torchvision
from torchvision import transforms

# Vérifier les versions
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# Vérifier si GPU est disponible (devrait afficher False pour une installation CPU-only)
print("GPU disponible:", torch.cuda.is_available())

# Créer un tenseur PyTorch simple
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print("Tenseur x:", x)
print("Tenseur y:", y)
print("Résultat z:", z)

# Tester une transformation torchvision
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionner une image (simulé ici)
    transforms.ToTensor()          # Convertir en tenseur
])
print("Transformations créées avec succès.")

