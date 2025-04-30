import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
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

# Obter dados e labels
images = batch_1[b'data']
labels = batch_1[b'labels']

# Selecionar 10000 imagens do batch 1
indices = random.sample(range(len(images)), 10000)
images_subset = [images[i] for i in indices]
labels_subset = [labels[i] for i in indices]

# Transformação com redimensionamento para 32x32 (CIFAR-10 original size)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
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

# Criar dataset e dataloader com 10000 imagens
dataset = CIFAR10Dataset(images_subset, labels_subset, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

# Função para avaliar um modelo
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

# Usar GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Carregar e avaliar os modelos
models = {
    "MobileNetV2 (x0.5)": torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_mobilenetv2_x0_5', pretrained=True, trust_repo=True),
    "ResNet32": torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True, trust_repo=True),
    "VGG13 (com BN)": torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg13_bn', pretrained=True, trust_repo=True)
}

for model_name, model in models.items():
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Accuracy do modelo {model_name}: {accuracy:.2f}%")
