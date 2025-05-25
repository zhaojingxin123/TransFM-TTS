""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn
from einops import rearrange

import matcha.utils as utils
from matcha.utils.model import sequence_mask

log = utils.get_pylogger(__name__)

# 层归一化
# 段代码实现了一个简单的层归一化模块，它可以应用于任何形状的张量，只要确保第一个维度是批处理大小，
# 第二个维度是特征维度（通道数）。通过计算每个样本的特征均值和方差，并使用可学习的参数进行调整，层归一化有助于加速训练过程并改善模型的性能。
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    '''增强提取的能力'''
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))

        self.norm_layers.append(LayerNorm(hidden_channels))

        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))

        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        # x_mask:一个掩码张量，用于表示音频数据中的有效部分，形状为(batch_size, 1, seq_len)
        x_org = x
        for i in range(self.n_layers):
            # x和x_mask相乘，然后再提取
            #self.conv_layers =ModuleList((0-2): 3 x Conv1d(192, 192, kernel_size=(5,), stride=(1,), padding=(2,))) 
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

# 时长预测器
# 时长预测的结构是怎么样的？
class DurationPredictor(nn.Module):
    ''' '''
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels# 192输入特征数
        # 为什么filter_channels是768？
        self.filter_channels = filter_channels #
        self.p_dropout = p_dropout #

        self.drop = torch.nn.Dropout(p_dropout)#
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)
    # 
    def forward(self, x, x_mask):
        
        # print('时长预测模块')
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        # 最终通过卷积层将文本特征映射,到持续时间预测值上，得到每个文本片段的持续时间预测
        return x * x_mask

# 
class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        x = rearrange(x, "b h t d -> t b h d")

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    '''这段代码定义了一个名为 MultiHeadAttention 的类，它是一个多头注意力机制（Multi-Head Attention Mechanism）的实现。
    这种机制在 Transformer 模型中非常关键，因为它允许模型同时关注输入的不同位置，从而捕捉到输入序列中的多种依赖关系'''
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        heads_share=True,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    '''这段代码定义了一个名为 FFN 的类，它是一个前馈神经网络（Feed-Forward Network）模块，
    通常用于序列建模任务中。该模块主要包括两个一维卷积层和一个 Dropout 层，用于对输入数据进行非线性变换和特征提取'''
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        # print('FFN前馈')
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    # 实现了基于多头自注意力机制（Multi-Head Self-Attention）和前馈神经网络（Feed-Forward Network）的编码器。
    # 这种结构通常用于 Transformer 模型中，适用于处理序列数据，如自然语言处理或语音处理任务。
    def __init__(
        self,
        hidden_channels,#隐藏层的通道数。
        filter_channels,#前馈神经网络的滤波器通道数。
        n_heads,#多头注意力机制的头数。
        n_layers,#编码器的层数。
        kernel_size=1,#卷积核的大小。
        p_dropout=0.0,#dropout 的概率。
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        
        self.attn_layers = torch.nn.ModuleList()#多头注意力机制层列表。
        self.norm_layers_1 = torch.nn.ModuleList()#包含多个 LayerNorm 层，用于多头注意力后的归一化
        self.ffn_layers = torch.nn.ModuleList()#包含多个前馈神经网络层。
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            # 添加多头注意力机制
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        # TextEncoder里面的Encoder
        '''在每层编码器中，首先通过多头自注意力机制处理输入，保留了序列中的依赖关系，并通过残差连接和归一化增强了模型的表现力。
            接着通过前馈神经网络进一步提取特征，并再次通过残差连接和归一化。
            最终的输出通过掩码操作，确保无效位置的数据被置零。'''
        # print('text——encoder里面的一个小的前向encoder')
        
        # attn_mask：根据 x_mask 生成注意力掩码，形状为 (batch_size, seq_length, seq_length)。
        # 这个掩码用于多头注意力机制中，确保模型仅关注有效位置。
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        for i in range(self.n_layers):
            # 应用掩码 x_mask 到输入 x
            x = x * x_mask

            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        # 最终返回的 x 是经过多层编码器处理后的结果
        # print('text——encoder里面的一个小的前向encoder结束',x.shape)
        return x

# 在这里加上一个，输入emo的分类，对进行
class TextEncoder(nn.Module):
    """Run forward pass to the transformer based encoder and duration predictor

    Args:
        x (torch.Tensor): text input
            shape: (batch_size, max_text_length)
        x_lengths (torch.Tensor): text input lengths
            shape: (batch_size,)
        spks (torch.Tensor, optional): speaker ids. Defaults to None.
            shape: (batch_size,)

    Returns:
        mu (torch.Tensor): average output of the encoder
            shape: (batch_size, n_feats, max_text_length)
        logw (torch.Tensor): log duration predicted by the duration predictor
            shape: (batch_size, 1, max_text_length)

        x_mask (torch.Tensor): mask for the text input
            shape: (batch_size, 1, max_text_length)
    """
    def __init__(
        self,
        encoder_type,
        encoder_params,
        duration_predictor_params,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        # 表示词汇表的大小，也就是嵌入层可以处理的最大索引数。每个索引对应词汇表中的一个词条
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        # 表示每个词条的嵌入向量的维度大小，也就是输出向量的长度
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks
        # 这里是映射到一个空间当中去
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        # --------------emo的映射函数-----------------------------

        # 这一部分就可以把e——vits的东西映射进来了
        # 映射
        # self.emo_proj = nn.Linear(1024, self.n_channels)

        # -----------------------------------------------------------

        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)

        # 这个mask是干嘛的？，前处理网络，对音素的id进行编码
        #   prenet: true
        # 这里的n_layers是3层是固定的
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x


        # TextEncoder里面的Encoder，这里是对音速进行初步的编码提取
        self.encoder = Encoder(
            encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )



        # 通过将说话人嵌入与基本特征拼接在一起，然后通过一个卷积层进行变换，可以使模型学习到如何根据说话人信息调整生成的语音特征。
        # self.n_channels + (spk_emb_dim if n_spks > 1 else 0)输入维度，
        # self.n_feats输出维度，变化后的特征维度
        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)

        # 时长预测
        self.proj_w = DurationPredictor(
            # 这是 DurationPredictor 的输入通道数。
            # 如果 n_spks 大于 1，则输入通道数为 self.n_channels 加上 spk_emb_dim；否则，输入通道数仅为 self.n_channels
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): log duration predicted by the duration predictor
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        # print('TEXTencoder前向')
        # print(f'输入到textencoder的x：{x.shape}')

        # print(f'self.n_channels的储存位置：{self.n_channels.device}')
        # 
        x = self.emb(x) * math.sqrt(self.n_channels)
        # print("textencoderd的emb后的x，变成了192维度--x", x.shape)
        # --------------emo的映射函数然后融合推理时和训练时的不一样-----------------------------
        # #
        # emos=emos.to("cuda")
        # x=x+self.emo_proj(emos.unsqueeze(0))
        # ---------------------------训练时可以用------------------------------------
        # 提取的emos的维度是B*1*1024
        # emos=emos.to("cuda")
        # e=self.emo_proj(emos.unsqueeze(1))

        # 这里是简单相加肯定是不行的，因为维度不一样，所以这里需要用卷积层
        # 这里似乎用拼接也可以，sovitssvc里面的f0是怎么使用的？
        # x=x+e
        
        
        # print("textencoderd的结束输出--x", x.shape)
        # ---------------------------------------------------------------
        x = torch.transpose(x, 1, -1)
        # 掩码的生成矩阵
        # sequence_mask(x_lengths, x.size(2)使用x_lengths生成一个掩码矩阵
        # 增加1个维度,抓换为和x一样的长度
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        # 使用掩码和x相乘，将x中对应掩码为0的位置置为0，并用三层cov1d提取
        x = self.prenet(x, x_mask)

        # 维度变成了4,256,105，B*CH*L
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        # x_mask用于标记序列中的有效位置
        # text_encoder内部的encoder
        # 这种结构通常用于 Transformer 模型中，使用Transformer的encoder对输入序列进行编码
        x = self.encoder(x, x_mask)#[B,256,最大长度]

        # 把x也就是输入的文本编码，映射到mu,mel频谱的初步预测？？？？提取以后的80维的特征，对应后面的mel频谱作为mu，使用的是一维卷积，长度不变
        mu = self.proj_m(x) * x_mask

        # detach()函数用于从计算图中分离张量，返回一个新的张量，该张量的梯度不会被计算
        x_dp = torch.detach(x)

        # 使用时长预测模块，时长预测的输出维度是logw
        logw = self.proj_w(x_dp, x_mask)
        # mu:是text的特征,logw:时长预测,这里的时长改变,可以加一些东西 x_mask:掩码
        # print("textencoderd的结束输出")
        # print("textencoderd的结束输出--mu", mu.shape)
        # print("textencoderd的结束输出--logw", logw.shape)
        # print("textencoderd的结束输出--x_mask", x_mask.shape)
        return mu, logw, x_mask
