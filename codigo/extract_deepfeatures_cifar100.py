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

parser = argparse.ArgumentParser(description="Extrair deep features do CIFAR-100.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset CIFAR-100.")
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

# Carregar os dados do CIFAR-100
meta = unpickle(os.path.join(data_dir, 'meta'))  # Informações sobre as classes
train_data = unpickle(os.path.join(data_dir, 'train'))  # Dados de treino
test_data = unpickle(os.path.join(data_dir, 'test'))    # Dados de teste

train_images = train_data[b'data']
train_labels = np.array(train_data[b'fine_labels'])
test_images = test_data[b'data']
test_labels = np.array(test_data[b'fine_labels'])

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
        # Converte os dados de 1D para 3 canais RGB
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

# dataset = CustomCIFARDataset(images, labels, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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
# Função para extrair deep features de um conjunto de imagens e labels
def extract_features(images, labels, model, device, batch_size=32):
    dataset = CustomCIFARDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.extend(targets.numpy())
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    return features, labels

# Extrair e salvar deep features de treino
print("Extraindo deep features de treino...")
train_features, train_labels = extract_features(train_images, train_labels, model, device)
output_train = os.path.join(deepfeatures_dir, "cifar100_deepfeatures_train.pkl")
with open(output_train, "wb") as f:
    pickle.dump({"features": train_features, "labels": train_labels}, f)
print(f"Deep features de treino salvas em: {output_train}")

# Extrair e salvar deep features de teste
print("Extraindo deep features de teste...")
test_features, test_labels = extract_features(test_images, test_labels, model, device)
output_test = os.path.join(deepfeatures_dir, "cifar100_deepfeatures_test.pkl")
with open(output_test, "wb") as f:
    pickle.dump({"features": test_features, "labels": test_labels}, f)
print(f"Deep features de teste salvas em: {output_test}")
