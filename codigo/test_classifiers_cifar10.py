import os
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time  # Importado para medir o tempo

# ----------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------
parser = argparse.ArgumentParser(description="Testar classificadores no CIFAR-10 usando deep features.")
parser.add_argument("-f", "--features_file", type=str, required=True, help="Caminho para o arquivo de deep features.")
args = parser.parse_args()

features_file = args.features_file
if not os.path.exists(features_file):
    print(f"Erro: O arquivo de deep features '{features_file}' não existe.")
    exit(1)

graphics_dir = os.path.join(os.getcwd(), "graficos")
os.makedirs(graphics_dir, exist_ok=True)

# ----------------------------
# CARREGAR DEEP FEATURES
# ----------------------------
with open(features_file, "rb") as f:
    data = pickle.load(f)

X = data["features"]
y = data["labels"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
training_times = {}  # Dicionário para armazenar os tempos de treinamento

for name, clf in classifiers.items():
    print(f"Treinando {name}...")
    start_time = time.time()  # Início do tempo de treinamento
    clf.fit(X_train, y_train)
    end_time = time.time()  # Fim do tempo de treinamento
    training_time = end_time - start_time
    training_times[name] = training_time  # Armazena o tempo de treinamento

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}, Tempo de Treinamento: {training_time:.2f} segundos")

# ----------------------------
# PLOTAR RESULTADOS
# ----------------------------
plt.figure(figsize=(12, 6))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.bar(accuracies.keys(), accuracies.values(), color="skyblue")
plt.xlabel("Classificador")
plt.ylabel("Acurácia")
plt.title("Comparação de Acurácia dos Classificadores (CIFAR-10)")
plt.xticks(rotation=45)

# Gráfico de Tempo de Treinamento
plt.subplot(1, 2, 2)
plt.bar(training_times.keys(), training_times.values(), color="lightgreen")
plt.xlabel("Classificador")
plt.ylabel("Tempo de Treinamento (s)")
plt.title("Tempo de Treinamento dos Classificadores (CIFAR-10)")
plt.xticks(rotation=45)

plt.tight_layout()

output_file = os.path.join(graphics_dir, "cifar10_classifiers_comparison.png")
plt.savefig(output_file)
print(f"Gráfico salvo em: {output_file}")
