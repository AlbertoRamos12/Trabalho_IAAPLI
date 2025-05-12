import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse  # Adicionado para argumentos de linha de comando
import time  # Adicionado para medir o tempo

# ----------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------
# Verificar se CUDA está disponível
if not torch.cuda.is_available():
    print("CUDA não está disponível. Certifique-se de que uma GPU está configurada corretamente.")
    exit(1)

device = torch.device("cuda")
print(f"Usando dispositivo: {device}")

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Treinamento de uma CNN simples no CIFAR-10.")
parser.add_argument("-b", "--batches", type=int, default=1, help="Número de batches de treino a usar (1-5).")
parser.add_argument("-e", "--epochs", type=int, default=3, help="Número de épocas para o treinamento.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset CIFAR-10.")
args = parser.parse_args()

# Validar número de batches
num_batches = max(1, min(args.batches, 5))  # Garante que está entre 1 e 5
print(f"Usando {num_batches} batch(es) para o treinamento.")

# Validar número de épocas
num_epochs = max(1, args.epochs)  # Garante que o número de épocas seja pelo menos 1
print(f"Treinando por {num_epochs} época(s).")

# Diretório do dataset
data_dir = args.data_dir
if not os.path.exists(data_dir):
    print(f"Erro: O diretório do dataset '{data_dir}' não existe.")
    exit(1)

# Diretório para salvar modelos
model_dir = os.path.join(os.getcwd(), "modelos")
os.makedirs(model_dir, exist_ok=True)

# Diretório para salvar gráficos
graphics_dir = os.path.join(os.getcwd(), "graficos")
os.makedirs(graphics_dir, exist_ok=True)

# Número de classes (CIFAR-10 tem 10 classes)
num_classes = 10

# ----------------------------
# FUNÇÕES AUXILIARES
# ----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Carregar os batches do CIFAR-10
images, labels = [], []
for i in range(1, num_batches + 1):  # Carrega apenas o número de batches especificado
    batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
    images.append(batch[b'data'])
    labels.extend(batch[b'labels'])

images = np.vstack(images)  # Combina os batches em um único array
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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# DEFINIÇÃO DE UMA CNN SIMPLES
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

model = SimpleCNN().to(device)

# ----------------------------
# TREINO SIMPLES
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Iniciando treino...")
train_losses = []
train_accuracies = []

# Medir o tempo total de treinamento
start_time = time.time()

for epoch in range(num_epochs):  # Loop de épocas sem barra de progresso
    epoch_start_time = time.time()  # Tempo de início da época
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), torch.tensor(targets).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcular acurácia
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Tempo da época
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Print ao final de cada época
    print(f"Época {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Tempo: {epoch_duration:.2f} segundos")

# Tempo total de treinamento
end_time = time.time()
total_duration = end_time - start_time
print(f"Treino concluído em {total_duration:.2f} segundos.")

# Salvar o modelo treinado
model_filename = f"simple_cnn_b{num_batches}_e{num_epochs}.pth"
model_path = os.path.join(model_dir, model_filename)
torch.save(model.state_dict(), model_path)
print(f"Modelo salvo em: {model_path}")

# ----------------------------
# GRÁFICO DE LOSS E ACCURACY
# ----------------------------
def plot_training_metrics(train_losses, train_accuracies, graphics_dir):
    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss por Época')
    plt.grid()
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy por Época')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # Salvar gráfico como PNG
    graph_filename = f"training_metrics_b{num_batches}_e{num_epochs}.png"
    graph_path = os.path.join(graphics_dir, graph_filename)
    plt.savefig(graph_path)
    print(f"Gráfico salvo em: {graph_path}")

# Salvar o gráfico após o treinamento
plot_training_metrics(train_losses, train_accuracies, graphics_dir)
