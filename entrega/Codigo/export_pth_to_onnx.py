import torch
import torch.nn as nn
import argparse
import os

# SimpleCNN sem Dropout
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
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

# SimpleCNN com Dropout (correcao overfitting)
class SimpleCNNDropout(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNNDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
class ComplexCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ComplexCNN com Dropout (correcao overfitting)
class ComplexCNNDropout(nn.Module):
    def __init__(self, num_classes=100):
        super(ComplexCNNDropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta um modelo .pth para .onnx")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--simples", action="store_true", help="Usa arquitetura simples (CIFAR-10)")
    group.add_argument("-c", "--complexa", action="store_true", help="Usa arquitetura complexa (CIFAR-100)")
    parser.add_argument("--pth", type=str, required=True, help="Caminho para o ficheiro .pth")
    parser.add_argument("--onnx", type=str, required=False, help="Caminho de saída para o ficheiro .onnx")
    parser.add_argument("--correcao_overfitting", action="store_true", help="Usa arquitetura com Dropout (correção de overfitting)")
    args = parser.parse_args()

    # Selecionar arquitetura
    if args.simples:
        if args.correcao_overfitting:
            model = SimpleCNNDropout()
        else:
            model = SimpleCNN()
        dummy_input = torch.randn(1, 3, 32, 32)
    elif args.complexa:
        if args.correcao_overfitting:
            model = ComplexCNNDropout()
        else:
            model = ComplexCNN()
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("Selecione -s (simples) ou -c (complexa)")

    model.load_state_dict(torch.load(args.pth, map_location="cpu", weights_only=True))
    model.eval()

    # Caminho de saída
    if args.onnx:
        onnx_path = args.onnx
    else:
        onnx_path = os.path.splitext(args.pth)[0] + ".onnx"

    # Exportar para ONNX
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
    print(f"Exportado para: {onnx_path}")
