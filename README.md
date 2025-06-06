# CIFAR Image Classification - IAAPLI Project

This repository contains the code and report for a project developed in the Applied Artificial Intelligence course (IAAPLI), focused on image classification using the CIFAR-10 and CIFAR-100 datasets. The work explores and compares deep learning (CNNs) and classical machine learning methods, including the use of feature extraction and dimensionality reduction.

## Project Overview

- **Datasets:** CIFAR-10 and CIFAR-100 (tiny RGB images, 10 and 100 classes respectively)
- **Models:** 
  - Custom Convolutional Neural Networks (Simple and Complex, with and without regularization)
  - Classical classifiers (Logistic Regression, Random Forest, KNN, SVM, MLP) trained on CNN features
  - Pre-trained models (ResNet-18, DenseNet-121, VGG-19) for benchmarking
- **Techniques:** Data augmentation, dropout, weight decay, PCA for feature reduction
- **Frameworks:** PyTorch, scikit-learn, matplotlib

## Repository Structure

- `codigo/`
  - `cnn_simples_10.py`, `cnn_simples_100.py`: Simple CNNs for CIFAR-10/100
  - `cnn_complexa_10.py`, `cnn_complexa_100.py`: Complex CNNs for CIFAR-10/100
  - `cnn_simples_10_correcao_overfitting.py`, etc.: CNNs with overfitting correction
  - `test_classifiers_cnn_simples_cifar10.py`: Classical classifiers on CNN features
  - `finetune_resnet.py`, `resnet18.py`: Transfer learning and pre-trained models
  - `carregar_dataset.py`: Dataset loading and visualization utilities
  - `template.tex`: Full project report (in English, with results, tables, and figures)
- `modelos/`: Saved model weights
- `graficos/`: Training curves, confusion matrices, and result plots

## How to Run

1. **Install dependencies:**
   - Python 3.8+
   - PyTorch, torchvision
   - scikit-learn
   - matplotlib
   - numpy, pillow

2. **Download CIFAR datasets:**
   - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
   - [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
   - Extract to a folder (e.g., `cifar_10_batches`, `cifar_100_batches`)

3. **Train a model (example for CIFAR-10 simple CNN):**
   ```bash
   python codigo/cnn_simples_10.py -d /path/to/cifar_10_batches -e 200
   ```

4. **Evaluate classical classifiers on CNN features:**
   ```bash
   python codigo/test_classifiers_cnn_simples_cifar10.py -d /path/to/cifar_10_batches -m modelos/simple_cnn_10_e200.pth
   ```

5. **See results:**
   - Training curves and confusion matrices are saved in `graficos/`
   - Model weights are saved in `modelos/`

## Main Results

- **Complex CNNs** outperform simple CNNs, especially on CIFAR-10.
- **Regularization** (dropout, data augmentation, weight decay) is essential to prevent overfitting.
- **Classical classifiers** perform better with features from deeper CNNs; PCA reduces training time with minimal accuracy loss.
- **Pre-trained models** (e.g., DenseNet-121) achieve >93% accuracy on CIFAR-10, far surpassing custom models.
- **CIFAR-100** remains challenging for custom models, highlighting the need for more advanced architectures.

See the full report in [`Image_Classification_CIFAR.pdf`](Image_Classification_CIFAR.pdf) for detailed methodology, results, tables, and discussion.

## Authors

- Alberto Ramos (1221165@isep.ipp.pt)
- Nuno Azevedo (1221167@isep.ipp.pt)

## License

This project is for academic purposes. See the report for references and dataset licenses.
