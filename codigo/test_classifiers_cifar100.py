import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------
parser = argparse.ArgumentParser(description="Testar classificadores no CIFAR-100 usando deep features extraídas de uma CNN.")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Caminho para o diretório do dataset CIFAR-100.")
parser.add_argument("-m", "--model", type=str, required=True, help="Caminho para o modelo .pth treinado da CNN.")
parser.add_argument("--pca_components", type=float, default=None, help="Número de componentes principais para PCA (int) ou fração de variância (float, ex: 0.95).")
args = parser.parse_args()

data_dir = args.data_dir
model_path = args.model

if not os.path.exists(data_dir):
    print(f"Erro: O diretório de dados '{data_dir}' não existe.")
    exit(1)
if not os.path.exists(model_path):
    print(f"Erro: O arquivo do modelo '{model_path}' não existe.")
    exit(1)

graphics_dir = os.path.join(os.getcwd(), "graficos")
os.makedirs(graphics_dir, exist_ok=True)

# ----------------------------
# DEFINIÇÃO DA CNN (sem última camada)
# ----------------------------
class SimpleCNN_Features(nn.Module):
    def __init__(self):
        super(SimpleCNN_Features, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.features(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Usando dispositivo para extração de features: {device} ({gpu_name})")
else:
    print(f"Usando dispositivo para extração de features: {device}")
model = SimpleCNN_Features().to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
state_dict = {k: v for k, v in state_dict.items() if "net.8" not in k and "net.9" not in k}
model.load_state_dict(state_dict, strict=False)
model.eval()

# ----------------------------
# FUNÇÃO PARA EXTRAIR FEATURES DE UM BATCH CIFAR-100
# ----------------------------
def extract_features_from_cifar(images, labels, model, device, batch_size=128):
    images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    tensor_images = torch.from_numpy(images).to(device)
    features = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_images), batch_size):
            batch = tensor_images[i:i+batch_size]
            feats = model(batch)
            features.append(feats.cpu().numpy())
    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels

# ----------------------------
# CARREGAR DADOS DO CIFAR-100
# ----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Carregar nomes das classes do ficheiro meta
meta = unpickle(os.path.join(data_dir, 'meta'))
cifar100_classes = [label.decode('utf-8') for label in meta[b'fine_label_names']]

train_data = unpickle(os.path.join(data_dir, 'train'))
test_data = unpickle(os.path.join(data_dir, 'test'))

images = train_data[b'data']
labels = np.array(train_data[b'fine_labels'])

# Split 80% treino, 20% validação (stratify para manter proporção de classes)
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

test_images = test_data[b'data']
test_labels = np.array(test_data[b'fine_labels'])

# ----------------------------
# EXTRAIR FEATURES
# ----------------------------
print("Extraindo deep features do treino...")
X_train, y_train = extract_features_from_cifar(train_images, train_labels, model, device)
print("Extraindo deep features da validação...")
X_val, y_val = extract_features_from_cifar(val_images, val_labels, model, device)
print("Extraindo deep features do teste...")
X_test, y_test = extract_features_from_cifar(test_images, test_labels, model, device)

# ----------------------------
# PCA (opcional)
# ----------------------------
pca_info = ""
if args.pca_components is not None:
    print(f"Aplicando PCA para {args.pca_components} componentes...")
    pca = PCA(n_components=args.pca_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    n_pca = X_train.shape[1]
    print(f"Número de componentes PCA usados: {n_pca}")
    pca_info = f"PCA: {n_pca}"
else:
    pca_info = "Sem PCA"

# ----------------------------
# TESTAR CLASSIFICADORES
# ----------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "SVM": SVC()
}

accuracies = {}
training_times = {}

for name, clf in classifiers.items():
    print(f"Treinando {name}...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    training_times[name] = training_time

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}, Tempo de Treinamento: {training_time:.2f} segundos")

    # Gerar e salvar matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cifar100_classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    if args.pca_components is not None:
        cm_filename = os.path.join(graphics_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}_pca{n_pca}.png")
    else:
        cm_filename = os.path.join(graphics_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}_sem_pca.png")
    plt.title(f"Matriz de Confusão - {name}\n{pca_info}")
    plt.savefig(cm_filename)
    plt.close()
    print(f"Matriz de confusão salva em: {cm_filename}")

# ----------------------------
# PLOTAR RESULTADOS
# ----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(accuracies.keys(), accuracies.values(), color="skyblue")
plt.xlabel("Classificador")
plt.ylabel("Acurácia")
plt.title(f"Comparação de Acurácia dos Classificadores (CIFAR-100)\n{pca_info}")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(training_times.keys(), training_times.values(), color="lightgreen")
plt.xlabel("Classificador")
plt.ylabel("Tempo de Treinamento (s)")
plt.title(f"Tempo de Treinamento dos Classificadores (CIFAR-100)\n{pca_info}")
plt.xticks(rotation=45)

plt.tight_layout()

if args.pca_components is not None:
    output_file = os.path.join(graphics_dir, f"cifar100_classifiers_comparison_pca{n_pca}.png")
else:
    output_file = os.path.join(graphics_dir, "cifar100_classifiers_comparison_sem_pca.png")
plt.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")
