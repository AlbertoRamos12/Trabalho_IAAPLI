import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# ----------------------------
# ARGUMENTOS DE LINHA DE COMANDO
# ----------------------------
parser = argparse.ArgumentParser(description="Testar modelo treinado em CIFAR-10 ou CIFAR-100.")
parser.add_argument("-m", "--model_path", type=str, required=True, help="Caminho para o arquivo .pth do modelo.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset.")
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], required=True, help="Tipo de dataset: cifar10 ou cifar100.")
args = parser.parse_args()

# ----------------------------
# CONFIGURAÇÕES
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# ----------------------------
# CARREGAR DADOS DE TESTE
# ----------------------------
if args.dataset == "cifar10":
    test_batch = unpickle(os.path.join(args.data_dir, "test_batch"))
    images = test_batch[b'data']
    labels = np.array(test_batch[b'labels'])
    num_classes = 10
elif args.dataset == "cifar100":
    test_batch = unpickle(os.path.join(args.data_dir, "test"))
    images = test_batch[b'data']
    labels = np.array(test_batch[b'fine_labels'])
    num_classes = 100
else:
    raise ValueError("Dataset inválido.")

# ----------------------------
# DEFINIÇÃO DO DATASET
# ----------------------------
class CustomCIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        r = img[0:1024].reshape(32, 32)
        g = img[1024:2048].reshape(32, 32)
        b = img[2048:].reshape(32, 32)
        img_rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
        if self.transform:
            img_rgb = self.transform(img_rgb)
        return img_rgb, self.labels[idx]

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomCIFARDataset(images, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# ----------------------------
# DEFINIÇÃO DO MODELO (necessário para carregar state_dict)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Instanciar o modelo e carregar os pesos
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# ----------------------------
# TESTE DO MODELO
# ----------------------------
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), torch.tensor(targets).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

avg_loss = test_loss / total
accuracy = 100 * correct / total

print(f"Resultados no conjunto de teste:")
print(f"Loss médio: {avg_loss:.4f}")
print(f"Acurácia: {accuracy:.2f}%")
