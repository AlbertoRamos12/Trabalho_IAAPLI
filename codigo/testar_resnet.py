import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import argparse

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_batches(data_dir, dataset='cifar10'):
    images = []
    labels = []
    if dataset == 'cifar10':
        # Usar apenas o batch de teste para avaliação
        batch = unpickle(os.path.join(data_dir, 'test_batch'))
        images = batch[b'data']
        labels = batch[b'labels']
        meta = unpickle(os.path.join(data_dir, 'batches.meta'))
        label_names = [l.decode('utf-8') for l in meta[b'label_names']]
    else:  # cifar100
        batch = unpickle(os.path.join(data_dir, 'test'))
        images = batch[b'data']
        labels = batch[b'fine_labels']
        meta = unpickle(os.path.join(data_dir, 'meta'))
        label_names = [l.decode('utf-8') for l in meta[b'fine_label_names']]
    return images, labels, label_names

def preprocess_images(images):
    # Normalizar como no PyTorch: [0,255] -> [0,1] -> normalizar
    images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    images = (images - 0.5) / 0.5  # Normalização [-1,1]
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weights', type=str, default=None, help='Caminho para pesos treinados em CIFAR (opcional)')
    args = parser.parse_args()

    print(f"Carregando {args.dataset} de {args.data_dir} ...")
    images, labels, label_names = load_cifar_batches(args.data_dir, args.dataset)
    images = preprocess_images(images)
    labels = np.array(labels)
    num_classes = 10 if args.dataset == 'cifar10' else 100

    # Ajustar a primeira camada convolucional da ResNet18 para aceitar 32x32
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()  # Remove maxpool para CIFAR
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if args.weights:
        print(f"Carregando pesos treinados de: {args.weights}")
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    else:
        print("Atenção: Não existem pesos oficiais da ResNet treinada em CIFAR no torchvision. Usando pesos aleatórios.")

    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    batch_size = 128
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = torch.from_numpy(images[i:i+batch_size]).float().to(device)
            batch_labels = torch.from_numpy(labels[i:i+batch_size]).long().to(device)
            outputs = net(batch_imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            if (i // batch_size) % 10 == 0:
                print(f"Processados {min(i+batch_size, len(images))} exemplos...")

    print(f"Acurácia no {args.dataset}: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
