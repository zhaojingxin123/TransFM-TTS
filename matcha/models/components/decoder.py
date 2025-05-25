import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from matcha.models.components.transformer import BasicTransformerBlock

# ？？？？是不是可以替换其他位置编码方式
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even，需要偶数"

    def forward(self, x, scale=1000):
        # print('SinusoidalPosEmb，时间嵌入')
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        # 计算得到一半的长度
        half_dim = self.dim // 2
        # 计算一个递减的指数序列 emb，其中每个元素都是通过 math.log(10000) 除以 half_dim - 1 的结果乘以 -1 
        # 再取指数得到的。这一步是为了产生不同频率的位置编码。
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        # 将输入 x 扩展一个新的维度，并与之前计算的 emb 张量相乘。这里的 unsqueeze 操作是为了确保广播机制能够正确应用。
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        # 将 emb 应用了 sin 和 cos 函数之后的结果拼接在一起，形成最终的位置嵌入。
        # 这里使用 torch.cat 沿着最后一个维度进行拼接。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )
    # 在这里使用的是等长卷积，只是改变了输入和输出的dim，还使用了GroupNorm
    def forward(self, x, mask):
        # print(' dec的Block1D前向')
        # 前向传播方法将首先将输入x和mask相乘，这里的dim是固定的
        # 然后将结果通过self.block处理，
        # 最后将结果和mask相乘形成输出。
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        # 初始化一维卷积的维度，第二个的
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        # print('dec的ResnetBlock1D前向')
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output # torch.Size([3, 256, 312])


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        # 减少为一般的长度，维度不变
        # print("torch.nn.Conv1d(dim, dim, 3, 2, 1) decoder的前向")
        return self.conv(x)

# 
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        # 定义第一个线性层，将输入的样本映射到时间嵌入维度
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)


        #CFM其中 条件的映射
        if cond_proj_dim is not None:

            # 如果有条件投影维度，则定义条件投影层
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            # 否则，条件投影层为None
            self.cond_proj = None

        # 定义激活函数
        self.act = get_activation(act_fn)

        # ？？？？
        if out_dim is not None:
            # 如果有输出维度，则将时间嵌入维度设置为输出维度
            time_embed_dim_out = out_dim
        else:
            # 否则，将时间嵌入维度设置为输入的时间嵌入维度
            time_embed_dim_out = time_embed_dim
        # 定义第二个线性层，将时间嵌入维度映射到输出维度
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            # 如果没有后处理激活函数，则后处理激活函数为None
            self.post_act = None
        else:
            # 否则，定义后处理激活函数
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        # print("TimestepEmbedding，时间步嵌入")
        if condition is not None:
            # 如果有条件，则将条件投影到样本上
            sample = sample + self.cond_proj(condition)
        # 将样本映射到时间嵌入维度
        sample = self.linear_1(sample)

        if self.act is not None:
            # 如果有激活函数，则对样本进行激活
            sample = self.act(sample)

        # 将时间嵌入维度映射到输出维度
        sample = self.linear_2(sample)

        if self.post_act is not None:
            # 如果有后处理激活函数，则对样本进行后处理激活
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
            
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        # print("这是在U-NET里面了，decoder的A 1D upsampling layer with an optional convolution.")
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")#如果不使用转置卷积，则首先使用 F.interpolate 函数对输入进行上采样，放大两倍（scale_factor=2.0），插值模式为最近邻插值（mode="nearest"）。

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        # print("ConformerWrapper")
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    '''U-net结构的估计器，使用U-net进行 '''
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        super().__init__()
        # channels是（256,256）
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        #时间嵌入
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        
        time_embed_dim = channels[0] * 4
        # 时间步的嵌入
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # 这是U-NET的三个块
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        
        #这里是创建resnet+attention+downsample的下采样块
        # 这个输入的维度是mel的80维度吗？要有256个channel吗？用了两边downsampleblock
        # print("len(channels)",len(channels))
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            
            is_last = i == len(channels) - 1
            # 创建ResnetBlock1D对象，用于处理输入和输出通道
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            # 创建一个ModuleList对象，用于存储TransformerBlock对象，还可以使用n_block选择使用几个transformer块
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            # 如果不是最后一个通道，则使用Downsample1D对象进行下采样，否则使用nn.Conv1d对象进行卷积,311是原长度
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            #  将resnet、transformer_blocks和downsample对象添加到down_blocks列表中
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))


        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)

        for i in range(len(channels) - 1):
            
            input_channel = channels[i]
            output_channel = channels[i + 1]
            
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 循环输入的是X，预测完成的一个Xt，就会再次在X位置输入，
    # 使用的spk，和mu就会引导合成不同的语音，其实可以把spk换成spk_emb
    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
           
            x (torch.Tensor): 目标mel，shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            mu  ：产生出来的mel
            t (_type_): shape (batch_size)

            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.

            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # print("这是一个Decoder Forward 输入的数据是：：")
        # print("x.shape",x.shape)
        # print("mu.shape",mu.shape)
        # print("t.shape",t.shape)
        # print("mask.shape",mask.shape)
        # print("spks.shape",spks.shape)
        
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        # 文本encoder出来的mu，这是为了适应后续的Transformer模块的输入格式。
        x = pack([x, mu], "b * t")[0]
        # print("进入CFM的数据x.shape",x.shape)
        
        # 对应的产生spk的向量
        # t维度的意思是时间长度？？？
        
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        # print("进入CFM的数据shape",x.shape)
        # print("进入CFM的数据shape，当有Xshape时",x.shape)
        
        hiddens = []
        masks = [mask]
        # print("mask出有效部分",masks)
        for resnet, transformer_blocks, downsample in self.down_blocks:
            
            mask_down = masks[-1]
            # 首先是进一个resnet的模型结构
            x = resnet(x, mask_down, t) #x:torch.Size([3, 224, 312]) mask_down： torch.Size([3, 1, 312]) t:torch.Size([3, 1024])
            # torch.Size([3, 256, 312])
            x = rearrange(x, "b c t -> b t c") #torch.Size([3, 312, 256])
            
            mask_down = rearrange(mask_down, "b 1 t -> b t")#torch.Size([3, 312])
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            
            x = downsample(x * mask_down)
            
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            
            mask_up = masks.pop()#从 masks 列表中弹出最后一个元素，并将其赋值给 mask_up。这通常用于获取当前层的掩码。
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)#pack([x, hiddens.pop()], "b * t")：将当前输入 x 和从 hiddens 列表中弹出的隐藏状态拼接在一起，并重新排列成形状为 (batch_size, *, time_steps) 的张量。
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
                
            x = rearrange(x, "b t c -> b c t")
            
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        # 这里出来的是x1
        return output * mask
