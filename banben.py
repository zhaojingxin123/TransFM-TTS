import torch
print(torch.__version__)  # 输出主版本号（如 2.1.0）
print(torch.version.cuda) # 输出编译时绑定的 CUDA 版本（如 11.8）