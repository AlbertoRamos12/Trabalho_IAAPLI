import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import random

# Caminho para a pasta onde estão os batches do CIFAR-10
data_dir = 'cifar_10_batches'

# Função para carregar ficheiros pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Carregar os arquivos do CIFAR-10
batch_1_path = os.path.join(data_dir, 'data_batch_1')
batch_1 = unpickle(batch_1_path)

# Carregar os nomes das classes
meta = unpickle(os.path.join(data_dir, 'batches.meta'))
label_names = [label.decode('utf-8') for label in meta[b'label_names']]

# Mostrar todas as classes e respetivos índices
print("Classes do CIFAR-10:")
for idx, name in enumerate(label_names):
    print(f"  {idx}: {name}")
print("\n")

# Obter dados e labels
images = batch_1[b'data']
labels = batch_1[b'labels']

# Selecionar 100 imagens aleatórias
indices = random.sample(range(len(images)), 100)
images_subset = [images[i] for i in indices]
labels_subset = [labels[i] for i in indices]

# Transformação com redimensionamento para 224x224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset customizado
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        r = img[0:1024].reshape(32, 32)
        g = img[1024:2048].reshape(32, 32)
        b = img[2048:].reshape(32, 32)
        img_rgb = np.stack([r, g, b], axis=2).astype(np.uint8)

        if self.transform:
            img_rgb = self.transform(img_rgb)

        return img_rgb, label

# Criar dataset e dataloader com 100 imagens aleatórias
dataset = CIFAR10Dataset(images_subset, labels_subset, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

# Carregar modelo ResNet18 pré-treinado
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10)
model.eval()

# Usar GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
model.to(device)

# Avaliar e contar acertos
correct = 0
total = 0
img_idx = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        for i in range(inputs.size(0)):
            real_class = label_names[labels[i]]
            predicted_class = label_names[predicted[i]]
            print(f"Imagem {img_idx + 1} - real: {real_class} | previsto: {predicted_class}")
            img_idx += 1
            total += 1
            if predicted[i] == labels[i]:
                correct += 1

# Mostrar acurácia
accuracy = 100.0 * correct / total
print(f"\nPercentagem: {accuracy:.2f}% ({correct}/{total} corretas)")
