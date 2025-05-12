import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# ----------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------
if not torch.cuda.is_available():
    print("CUDA não está disponível. Certifique-se de que uma GPU está configurada corretamente.")
    exit(1)

device = torch.device("cuda")
print(f"Usando dispositivo: {device}")

parser = argparse.ArgumentParser(description="Extrair deep features do CIFAR-10.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset CIFAR-10.")
args = parser.parse_args()

data_dir = args.data_dir
if not os.path.exists(data_dir):
    print(f"Erro: O diretório do dataset '{data_dir}' não existe.")
    exit(1)

deepfeatures_dir = os.path.join(os.getcwd(), "deepfeatures")
os.makedirs(deepfeatures_dir, exist_ok=True)

# ----------------------------
# FUNÇÕES AUXILIARES
# ----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Carregar os batches do CIFAR-10
images, labels = [], []
for i in range(1, 6):  # CIFAR-10 tem 5 batches de treino
    batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
    images.append(batch[b'data'])
    labels.extend(batch[b'labels'])

images = np.vstack(images)
labels = np.array(labels)

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ----------------------------
# DEFINIÇÃO DE UMA CNN SIMPLES
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.feature_extractor(x)

model = SimpleCNN().to(device)
model.eval()

# ----------------------------
# EXTRAIR DEEP FEATURES
# ----------------------------
print("Extraindo deep features...")
deep_features = []
deep_labels = []

with torch.no_grad():
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        features = model(inputs)
        deep_features.append(features.cpu().numpy())
        deep_labels.extend(targets.numpy())

deep_features = np.vstack(deep_features)
deep_labels = np.array(deep_labels)

# Salvar deep features em um arquivo
output_file = os.path.join(deepfeatures_dir, "cifar10_deepfeatures.pkl")
with open(output_file, "wb") as f:
    pickle.dump({"features": deep_features, "labels": deep_labels}, f)

print(f"Deep features salvas em: {output_file}")
