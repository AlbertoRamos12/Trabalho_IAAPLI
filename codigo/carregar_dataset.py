import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('TkAgg')  # Usa o backend "TkAgg" para renderização gráfica com GUI

# Caminho para a pasta onde estão os batches
data_dir = '../cifar_10_batches'

# Função para carregar ficheiros pickle
def unpickle(file):
    print(f"A abrir o ficheiro: {file}")
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    print(f"Ficheiro {file} carregado com sucesso!")
    return dict

# Carregar o ficheiro data_batch_1
batch_1_path = os.path.join(data_dir, 'data_batch_1')
batch_1 = unpickle(batch_1_path)

# Carregar os nomes das classes
meta = unpickle(os.path.join(data_dir, 'batches.meta'))
label_names = [label.decode('utf-8') for label in meta[b'label_names']]

# Obter dados e labels
images = batch_1[b'data']
labels = batch_1[b'labels']

# Ver o formato das imagens e labels
print(f"Formato das imagens: {images.shape}")
print(f"Total de imagens no batch: {len(images)}")
print(f"Exemplo de labels: {labels[:5]}")  # Mostrar os primeiros 5 labels

# Escolher índice aleatório
idx = random.randint(0, len(images) - 1)
img = images[idx]
label = labels[idx]

# Mostrar qual imagem foi selecionada
print(f"Índice da imagem selecionada: {idx}")
print(f"Label da imagem: {label} - {label_names[label]}")

# Reconstruir a imagem RGB
r = img[0:1024].reshape(32, 32)
g = img[1024:2048].reshape(32, 32)
b = img[2048:].reshape(32, 32)
img_rgb = np.stack([r, g, b], axis=2)

# Mostrar imagem com nome e número da classe
print("Mostrando imagem...")
plt.imshow(img_rgb)
plt.title(f"Label: {label} - {label_names[label]}")
plt.axis('off')
plt.show()

