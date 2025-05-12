import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import time

# ----------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------
# Verifica se CUDA está disponível para usar a GPU
if not torch.cuda.is_available():
    print("CUDA não está disponível. Certifique-se de que uma GPU está configurada corretamente.")
    exit(1)

device = torch.device("cuda")
print(f"Usando dispositivo: {device}")

# Configuração de argumentos de linha de comando
parser = argparse.ArgumentParser(description="Treinamento de uma CNN simples no CIFAR-100.")
parser.add_argument("-e", "--epochs", type=int, default=3, help="Número de épocas para o treinamento.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset CIFAR-100.")
args = parser.parse_args()

# Número de épocas para o treinamento
num_epochs = max(1, args.epochs)
print(f"Treinando por {num_epochs} época(s).")

# Verifica se o diretório do dataset existe
data_dir = args.data_dir
if not os.path.exists(data_dir):
    print(f"Erro: O diretório do dataset '{data_dir}' não existe.")
    exit(1)

# Diretórios para salvar modelos e gráficos
model_dir = os.path.join(os.getcwd(), "modelos")
os.makedirs(model_dir, exist_ok=True)

graphics_dir = os.path.join(os.getcwd(), "graficos")
os.makedirs(graphics_dir, exist_ok=True)

# CIFAR-100 tem 100 classes
num_classes = 100

# ----------------------------
# FUNÇÕES AUXILIARES
# ----------------------------
# Função para carregar arquivos pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Carregar os dados do CIFAR-100
meta = unpickle(os.path.join(data_dir, 'meta'))  # Informações sobre as classes
train_data = unpickle(os.path.join(data_dir, 'train'))  # Dados de treino
test_data = unpickle(os.path.join(data_dir, 'test'))  # Dados de teste

# Imagens e rótulos do conjunto de treino
images = train_data[b'data']
labels = np.array(train_data[b'fine_labels'])

# ----------------------------
# DEFINIÇÃO DO DATASET
# ----------------------------
# Classe personalizada para o dataset CIFAR-100
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

        # Aplica transformações, se especificadas
        if self.transform:
            img_rgb = self.transform(img_rgb)

        return img_rgb, self.labels[idx]

# Transformação para converter imagens para tensores
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Criação do dataset e dataloader
dataset = CustomCIFARDataset(images, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------
# DEFINIÇÃO DE UMA CNN SIMPLES
# ----------------------------
# Definição da arquitetura da CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Camada convolucional 1
            nn.ReLU(),
            nn.MaxPool2d(2),  # Pooling 1

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Camada convolucional 2
            nn.ReLU(),
            nn.MaxPool2d(2),  # Pooling 2

            nn.Flatten(),  # Flatten para entrada na camada totalmente conectada
            nn.Linear(32 * 8 * 8, 128),  # Camada totalmente conectada 1
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Camada de saída
        )

    def forward(self, x):
        return self.net(x)

# Instancia o modelo e move para o dispositivo (GPU)
model = SimpleCNN().to(device)

# ----------------------------
# TREINO SIMPLES
# ----------------------------
# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Iniciando treino...")
train_losses = []
train_accuracies = []

# Medir o tempo total de treinamento
start_time = time.time()

# Loop de treinamento por época
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop sobre os batches
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), torch.tensor(targets).to(device)

        optimizer.zero_grad()  # Zera os gradientes
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calcula a perda
        loss.backward()  # Backward pass
        optimizer.step()  # Atualiza os pesos

        running_loss += loss.item()

        # Calcula a acurácia
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calcula métricas da época
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(f"Época {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Tempo: {epoch_duration:.2f} segundos")

# Tempo total de treinamento
end_time = time.time()
total_duration = end_time - start_time
print(f"Treino concluído em {total_duration:.2f} segundos.")

# Salvar o modelo treinado
model_filename = f"simple_cnn_100_e{num_epochs}.pth"  # Nome do modelo com "100"
model_path = os.path.join(model_dir, model_filename)
torch.save(model.state_dict(), model_path)
print(f"Modelo salvo em (CIFAR-100): {model_path}")

# ----------------------------
# GRÁFICO DE LOSS E ACCURACY
# ----------------------------
# Função para plotar métricas de treinamento
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
    graph_filename = f"simple_cnn_training_metrics_100_e{num_epochs}.png"  # Nome do gráfico com "100"
    graph_path = os.path.join(graphics_dir, graph_filename)
    plt.savefig(graph_path)
    print(f"Gráfico salvo em (CIFAR-100): {graph_path}")

# Salvar o gráfico após o treinamento
plot_training_metrics(train_losses, train_accuracies, graphics_dir)
