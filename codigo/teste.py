import torch
print("Versão:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("GPU usada:", torch.cuda.get_device_name(0))
print("Número de GPUs disponíveis:", torch.cuda.device_count())
print("Número de núcleos CUDA:", torch.cuda.get_device_capability(0))
print("Memória total da GPU:", torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), "GB")
