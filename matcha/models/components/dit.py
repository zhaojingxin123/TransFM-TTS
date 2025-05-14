"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat

from x_transformers.x_transformers import RotaryEmbedding

# from model.modules import (
#     TimestepEmbedding,
#     ConvNeXtV2Block,
#     ConvPositionEmbedding,
#     DiTBlock,
#     AdaLayerNormZero_Final,
#     precompute_freqs_cis, get_pos_embed_indices,
# )
from .f5_modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis, get_pos_embed_indices,
)
# Text embedding

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: int['b nt'], seq_len, drop_text = False):
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text_len), value = 0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)
        text = text.long()
        text = self.text_embed(text) # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding

# class InputEmbedding(nn.Module):
#     def __init__(self, mel_dim, text_dim, out_dim):
#         super().__init__()
#         # 过一个线性层变成out——dim
#         self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        
#         self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

#     def forward(self, x: float['b n d'], cond: float['b n d'], text_embed: float['b n d'], drop_audio_cond = False):
#         if drop_audio_cond:  # cfg for cond audio
#             cond = torch.zeros_like(cond)
#         # 简单的拼接
#         print(f"x shape: {x.shape}")
#         print(f"cond shape: {cond.shape}")
#         print(f"text_embed shape: {text_embed.shape}")
        
#         x = self.proj(torch.cat((x, cond, text_embed), dim = -1))
#         x = self.conv_pos_embed(x) + x
#         return x
    
    
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        # 过一个线性层变成out_dim，把text去掉 
        self.proj = nn.Linear(mel_dim*2+ text_dim, out_dim)
        
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], text_embed: float['b n d'], drop_audio_cond = False):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
        # 简单的拼接,去掉text，加上cond，这个cond作为mu
        # print(f"x shape: {x.shape}")
        # print(f"cond shape: {cond.shape}")
        # print(f"text_embed shape: {text_embed.shape}")
        
        # x = self.proj(torch.cat((x, cond, text_embed), dim = -1))
        # 把cond去掉了
        # 尝试去掉位置信息
        x = self.proj(torch.cat((x, cond,text_embed), dim = -1))
        x = self.conv_pos_embed(x) + x
        return x
    

# Transformer backbone using DiT blocks

class DiT(nn.Module):
    def __init__(self, *, 
                 dim, depth = 8, heads = 8, dim_head = 64, dropout = 0.1, ff_mult = 4,
                 mel_dim = 80, text_num_embeds = 256, text_dim = None, conv_layers = 0,
                 long_skip_connection = False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers = conv_layers)
        
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        # 长度进行了可变，每次都有seq——length
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias = False) if long_skip_connection else None
        
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        

    # forward(self, x, mask, mu, t, spks=None, cond=None):
    def forward(
        self,
        x: float['b n d'],     # nosied input audio
        mask: bool['b n'] ,
        cond: float['b n d'],  # 这里是mu
        time: float['b'] | float[''],  # time step
        text: int['b nt'],     # text这里其实是spk_emb
        drop_audio_cond,  # cfg for cond audio
        drop_text,        # cfg for text
       
    ):
        # print("Dit开始了")
        # 有一个可变长度的X
        mask = mask.squeeze(1)
        # print("x的形状",x.shape)
        # print("mask的形状",mask.shape)
        # print("mask的改变以后的形状",mask.shape)
        # print("cond的形状",cond.shape)
        # print("time的形状",time.shape)
        # print("text的形状",text.shape)
        # 交换x的第1和2维度，也交换cond的第1和2维度
        x = x.transpose(1,2)
        cond = cond.transpose(1,2)
        # print("x的形状",x.shape)
        # print("cond的形状",cond.shape)
        
        batch, seq_len = x.shape[0], x.shape[1]
        
        # print("batch",batch)
        # print("seq_len",seq_len)
        
        if time.ndim == 0:
            time = repeat(time, ' -> b', b = batch)
        # dit的forward部分
        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        # t=time
        # 文本也变成和mel一致的长度
        # 去掉这一部分
        # print("text的形状",text.shape)
        # text = self.text_embed(text, seq_len, drop_text = drop_text)
        # 直接就是一个拼接
        # print("text的形状",text.shape)
        text = text.unsqueeze(-1)
        text = text.repeat(1, 1, 80)
        # print("text的形状",text.shape)
        
        # 目标长度是 x 的时间维度长度
        target_length = x.size(1)

        # 使用线性插值调整 text_embed 的时间维度长度
        text = F.interpolate(text.permute(0, 2, 1), size=target_length, mode='linear').permute(0, 2, 1)
        # print("text的形状",text.shape)
        
        # 噪声音频。条件MU，text是spks_emb转化而来
        
        x = self.input_embed(x, cond, text, drop_audio_cond = drop_audio_cond)
        
        # 对这些进行位置编码
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x
        # 
        # print("mask的形状",mask.shape)
        for block in self.transformer_blocks:
            # print("mask",mask)
            x = block(x, t, mask = mask, rope = rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim = -1))
            
        # 最后一个adaLN
        x = self.norm_out(x, t)
        # 最后就是通过一个mel进行维度的匹配
        output = self.proj_out(x)
        output = output.transpose(1,2)
        # 最后设计成输出一个【b,dim，mel-length】
        # print("output的形状",output.shape)
        return output
