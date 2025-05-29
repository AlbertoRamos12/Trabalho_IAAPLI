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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta um modelo .pth para .onnx")
    parser.add_argument("--pth", type=str, required=True, help="Caminho para o ficheiro .pth")
    parser.add_argument("--onnx", type=str, required=False, help="Caminho de saída para o ficheiro .onnx")
    parser.add_argument("--correcao_overfitting", action="store_true", help="Usa arquitetura com Dropout (correção de overfitting)")
    args = parser.parse_args()

    # Selecionar arquitetura
    if args.correcao_overfitting:
        model = SimpleCNNDropout()
    else:
        model = SimpleCNN()
    model.load_state_dict(torch.load(args.pth, map_location="cpu"))
    model.eval()

    # Caminho de saída
    if args.onnx:
        onnx_path = args.onnx
    else:
        onnx_path = os.path.splitext(args.pth)[0] + ".onnx"

    # Exportar para ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
    print(f"Exportado para: {onnx_path}")
