from abc import ABC

import torch
import torch.nn.functional as F
from matcha.utils.model import denormalize
# Matcha-TTS/matcha/cli_aishell_emo.py
# from matcha.cli_aishell_emo import to_waveform,load_vocoder

import soundfile as sf

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

import matplotlib.pyplot as plt
import numpy as np

from .dit import DiT




def plot_spectrogram_to_numpy(spectrogram, filename):
    # 创建一个图形和坐标轴，设置图形大小为12x3英寸
    fig, ax = plt.subplots(figsize=(12, 3))
    # 在坐标轴上绘制频谱图，设置长宽比为自动，原点在左下角，不使用插值
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    # 添加颜色条，用于表示频谱图的强度
    plt.colorbar(im, ax=ax)
    # 设置x轴标签为"Frames"，表示帧数
    plt.xlabel("Frames")
    # 设置y轴标签为"Channels"，表示通道数
    plt.ylabel("Channels")
    # 设置图形标题为"Synthesised Mel-Spectrogram"，表示合成的Mel频谱图
    plt.title("Synthesised Mel-Spectrogram")
    # 绘制图形，但不显示在屏幕上
    fig.canvas.draw()
    # 将绘制的图形保存为文件，文件名为传入的filename参数
    plt.savefig(filename)




log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        # print('前向Forward diffusion cfm')
        # 2025年3月24日08:04:34
        z = torch.randn_like(mu) * temperature
        # 改动起始位置2025年3月24日08:04:50
        # z=mu
        # 保存进入cfm之前的东西############这里是打印保存#########
        # plot_path='/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu.png'
        # plot_spectrogram_to_numpy(np.array(mu.squeeze().float().cpu()), plot_path)
        
        #   mel_mean: -6.699476
        #   mel_std: 2.451259

        # mel_mean=-6.699476
        # mel_std=2.451259
        
        # mu_denormlize = denormalize(mu, mel_mean,mel_std)
        
        # torch.save(mu_denormlize,'/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu_denormlize.pt')
        # torch.save(mu,'/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu.pt')
        # plot_path='/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu_denormlize.png'
        # plot_spectrogram_to_numpy(np.array(mu_denormlize.squeeze().float().cpu()), plot_path)
        
        
        #####################################------------------------#######################################
        # vocoder='hifigan_univ_v1'
        # path_vocoder_path='/home/zjx/Matcha-TTS/checkpoint/hifigan_univ_v1'
        # vocoder, denoiser = load_vocoder(vocoder, path_vocoder_path, device=torch.device("cuda"))
        # audio = vocoder(mu).clamp(-1, 1)
        # wav=audio.cpu().squeeze()
        
        # sf.write(f"/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/cfm_zhiqian.wav", wav, 22050, "PCM_24")
        # 把过程扩散成几步[0,0.01,0.02,0.03,0.04,0.05........0.999]
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # 在这里把mu画出来？ mu开始好，还是x开始好
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []


        # 这里可以测试到底是从mu开始还是从噪声开始
        for step in range(1, len(t_span)):
            # 使用Unet估计的速度
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond,drop_text=False)
            # print(f"采样学习过程中{step}步的mask的形状:",mask.shape)
            # dt也就是步长，时间步长
            x = x + dt * dphi_dt
            # 时间变为下一步的时间
            t = t + dt
            # print(f'这是第{step}步')
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            # 要想直接产生正常的mel，需要先denormlize
            
            #########################----------------每一步都进行输出.pt-----------------#######################################
            
            # mel_mean=-6.699476
            # mel_std=2.451259
            # x_denormlize = denormalize(x, mel_mean,mel_std)
            # torch.save(x_denormlize, f'/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/x_denormlize_{step}.pt')
            
            #########################---------------------------------#######################################
            
        # 返回最后一个x
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep随机的时间步，可不可以不随机？？？???
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)

        # sample noise p(x_0)，从标准正态分布中采样
        # z噪音原来的
        # z = torch.randn_like(x1)
        # 改动的改成从开始
        z=mu
        
        # z = torch.randn_like(x1)
        # y.shape=[4, 80, 408] 
        # z里面采样 x1 (torch.Tensor): Target
        # y是经过一步采样得到的,采样值Xt，此Xt也就是变量y，会作为下一步的输入继续进行估计
        # y=(1-t)*z+t*x1其实一样
        # 这里的z换成mu的话尺寸不一致，
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1


        # estimate noise p(x_0|y)估计噪声
        # u.shape=[4, 80, 408]
        # u是真实值减去一次采样的
        # u=x1 - 0.9999 * z
        # u应该等于
        
        # 这个是应该是什么样子的，如果是直线传输的话
        # u=x1 - 0.9999 * z
        # u = x1 -  z
        u = x1 - (1 - self.sigma_min) * z
        # ？？？？？

        # 这里使用了CFM
        # 去噪或者说分布转换后所得的U，与这里是cfm的输出，
        # mu (torch.Tensor): output of encoder shape: (batch_size, n_feats, mel_timesteps)
        # self.estimator(y, mask, mu, t.squeeze(), spks)这一步估计出来的噪声，和这一步应有的速度也就是噪声的差值
        drop_audio_cond=False
        drop_text=False
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks,drop_audio_cond,drop_text), u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        # print('计算CFM的loss')
        return loss, y

# 这里继承了上面的BASECFM
class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64,estimator_type="dit"):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        # 这里是U-net结构的decoder，使用采样进入u-net以后，将得到的是
        # self.estimator = Decoder是U-net结构,输入数据，输出数据和参数，输入的是 forward(self, x, mask, mu, t, spks=None, cond=None):
        # print("in_channels:{},out_channel:{}".format(in_channels,out_channel))
        # print('estimator_type:',estimator_type)
        
        if estimator_type == "matcha":
            # def forward(self, x, mask, mu, t, spks=None, cond=None):
            self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        
        
          
        elif estimator_type=="dit":
            # 最开始的配置22.1
            self.estimator =  DiT(dim=256, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月21日00:20:05结束的13.4
            # self.estimator =  DiT(dim=256, depth=2, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月21日09:17:21开始的配置模型时需要改动
            # self.estimator =  DiT(dim=128, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月21日22:46:19 这个是128D，depth=4 12M
            # self.estimator =  DiT(dim=128, depth=4, heads=8, dim_head=64, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月22日01:45:53 这个是128D，depth=4 dim_head=128 13M
            # self.estimator =  DiT(dim=128, depth=4, heads=8, dim_head=128, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月22日14:52:09 11.4参数
            # self.estimator =  DiT(dim=128, depth=4, heads=8, dim_head=32, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 2025年3月22日14:52:09 11.4参数
            # self.estimator =  DiT(dim=128, depth=2, heads=2, dim_head=32, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 20M
            # self.estimator =  DiT(dim=256, depth=8, heads=8, dim_head=32, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # 20M
            # self.estimator =  DiT(dim=128, depth=8, heads=8, dim_head=32, dropout=0.1, ff_mult=4, mel_dim=80, text_num_embeds=256, conv_layers=2, long_skip_connection=True)
            # self.estimator =  DiT(dim=in_channels, depth=decoder_params['depth'], heads=decoder_params['heads'], 
            #                      dim_head=decoder_params['dim_head'], dropout=decoder_params['dropout'], 
            #                      ff_mult=decoder_params['ff_mult'], mel_dim=out_channel, 
            #                      text_num_embeds=decoder_params['text_num_embeds'], 
            #                      text_dim=decoder_params['text_dim'], conv_layers=decoder_params['conv_layers'])
            