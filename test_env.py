import torch
import transformers
import sys
print(sys.executable)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print("显卡名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
print(f"Transformers版本: {transformers.__version__}")