""" from https://github.com/jaywalnut310/glow-tts """

import numpy as np
import torch

# 这个函数 sequence_mask 用于生成一个掩码矩阵（mask matrix），该矩阵可以用来标记序列中的有效部分和填充部分。
# 这对于处理变长序列（如文本、语音等）时非常有用，因为变长序列通常会被填充到相同的长度以便批量处理。
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# 这段代码定义了一个名为 fix_len_compatibility 的函数，它的目的是确保输入的长度 length 能够与 U-Net 架构中的下采样步骤兼容。
def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    # 
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    # 将 length 除以 factor，并对结果进行向上取整，然后乘以 factor。这样可以确保新的长度能够被 factor 整除。
    length = (length / factor).ceil() * factor
    # 进行整数化
    # print('目的是确保输入的长度 length 能够与 U-Net 架构中的下采样步骤兼容')
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape

# 要是提供dura的话就用这个
def generate_path(duration, mask):
    # 获取duration张量所在的设备（CPU或GPU）
    device = duration.device

    # 获取mask张量的形状，b为批次大小，t_x和t_y为时间步长和特征维度
    b, t_x, t_y = mask.shape
    # 计算累积持续时间，cum_duration[i, j]表示第i个样本在前j个时间步的累积持续时间 这就是那个指示函数
    cum_duration = torch.cumsum(duration, 1)
    # 初始化路径张量，形状为(b, t_x, t_y)，数据类型与mask相同，并放置在相同设备上
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    # 将累积持续时间张量展平为形状(b * t_x)
    cum_duration_flat = cum_duration.view(b * t_x)
    # 使用sequence_mask函数生成路径掩码，根据cum_duration_flat和t_y生成掩码
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    # 将路径掩码张量重新调整为形状(b, t_x, t_y)
    path = path.view(b, t_x, t_y)
    # 对路径掩码进行填充操作，填充形状为[[0, 0], [1, 0], [0, 0]]，即在第二个维度上从左侧填充一个0
    # 然后通过减去填充后的张量，实现路径的生成
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    # 将生成的路径与mask相乘，确保路径只在mask为1的位置有效
    path = path * mask
    # 返回生成的路径张量
    return path

# 函数的实现原理是，首先计算两个序列之间的时长差异（即(logw - logw_) ** 2），
# 然后将这个差异除以两个序列的长度（即torch.sum(lengths)），最后返回这个损失值。
def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss



def normalize(data, mu, std):
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std

# 
def denormalize(data, mu, std):
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu
