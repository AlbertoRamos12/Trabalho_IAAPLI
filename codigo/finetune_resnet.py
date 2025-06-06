import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import matplotlib.pyplot as plt  # Adicionado para gráficos

# ----------------------------
# ARGUMENTOS
# ----------------------------
parser = argparse.ArgumentParser(description="Fine-tuning da ResNet18 pré-treinada no ImageNet para CIFAR-10 ou CIFAR-100.")
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--data_dir', type=str, required=True, help='Diretório do dataset CIFAR')
parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--test_only', action='store_true', help='Apenas testa o modelo salvo, sem treinar')
parser.add_argument('--weights', type=str, default=None, help='Caminho para pesos salvos (.pth)')
args = parser.parse_args()

# ----------------------------
# CONFIGURAÇÕES
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

num_classes = 10 if args.dataset == 'cifar10' else 100

# ----------------------------
# FUNÇÕES AUXILIARES
# ----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

if args.dataset == 'cifar10':
    train_images, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(args.data_dir, f'data_batch_{i}'))
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    test_batch = unpickle(os.path.join(args.data_dir, 'test_batch'))
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
elif args.dataset == 'cifar100':
    train_batch = unpickle(os.path.join(args.data_dir, 'train'))
    test_batch = unpickle(os.path.join(args.data_dir, 'test'))
    train_images = train_batch[b'data']
    train_labels = np.array(train_batch[b'fine_labels'])
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'fine_labels'])

# ----------------------------
# DATASET
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

# Transforms: resize para 224x224, normalização do ImageNet
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_test = transform_train

train_dataset = CustomCIFARDataset(train_images, train_labels, transform=transform_train)
test_dataset = CustomCIFARDataset(test_images, test_labels, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ----------------------------
# MODELO
# ----------------------------
net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
net.fc = nn.Linear(net.fc.in_features, num_classes)
net = net.to(device)

# Congelar todas as camadas exceto a última
for param in net.parameters():
    param.requires_grad = False
for param in net.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.fc.parameters(), lr=args.lr)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

if args.weights:
    print(f"Carregando pesos do modelo: {args.weights}")
    net.load_state_dict(torch.load(args.weights, map_location=device))

if not args.test_only:
    # ----------------------------
    # TREINAMENTO
    # ----------------------------
    print("Iniciando fine-tuning...")
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            if not torch.is_tensor(targets):
                targets = torch.tensor(targets)
            else:
                targets = targets.clone().detach()
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Avaliação no teste a cada época
        net.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                if not torch.is_tensor(targets):
                    targets = torch.tensor(targets)
                else:
                    targets = targets.clone().detach()
                targets = targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        test_loss /= len(test_loader)
        test_acc = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Época {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Test_Loss: {test_loss:.4f}, Test_Acc: {test_acc:.2f}%, Tempo: {time.time()-start:.1f}s")

    # ----------------------------
    # SALVAR MODELO
    # ----------------------------
    model_dir = os.path.join(os.getcwd(), "modelos")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"finetuned_resnet18_{args.dataset}.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Modelo salvo em: {model_path}")

# Avaliação final (sempre executa se --test_only ou após treino)
net.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        else:
            targets = targets.clone().detach()
        targets = targets.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += targets.size(0)
        test_correct += (predicted == targets).sum().item()
test_acc = 100 * test_correct / test_total
print(f"Accuracy final no teste ({args.dataset}): {test_acc:.2f}%")

# ----------------------------
# GRÁFICO DE LOSS E ACCURACY
# ----------------------------
def plot_training_metrics(train_losses, train_accuracies, test_losses, test_accuracies, graphics_dir):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, marker='x', label='Test Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss por Época')
    plt.grid()
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Train Acc')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='x', label='Test Acc')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy por Época')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    graph_dir = os.path.join(os.getcwd(), "graficos")
    os.makedirs(graph_dir, exist_ok=True)
    graph_filename = f"finetune_resnet18_metrics_{args.dataset}_e{args.epochs}.png"
    graph_path = os.path.join(graph_dir, graph_filename)
    plt.savefig(graph_path)
    print(f"Gráfico salvo em: {graph_path}")

plot_training_metrics(train_losses, train_accuracies, test_losses, test_accuracies, os.getcwd())
