import torch
import soundfile as sf
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.hifigan.env import AttrDict
from matcha.hifigan.config import v1

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan



def load_vocoder(vocoder_name, checkpoint_path, device):
    print(f"[!] Loading {vocoder_name}!")
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder {vocoder_name} not implemented! define a load_<<vocoder_name>> method for it"
        )

    denoiser = Denoiser(vocoder, mode="zeros")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser
vocoder_name="hifigan_T2_v1"
vocoder_path="/home/zjx/Matcha-TTS/checkpoint/hifigan_univ_v1"
vocoder, denoiser = load_vocoder(vocoder_name,vocoder_path, device=torch.device("cuda:0"))


def to_waveform(mel, vocoder, denoiser=None):
    
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()

# mel_path='/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu_denormlize.pt'
mel_path='/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/x_denormlize_2.pt'
# mel_path='/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/mu_denormlize.pt'
mel=torch.load(mel_path)
wav_raw=to_waveform(mel, vocoder, denoiser)
wav_path="/home/zjx/Matcha-TTS/tts_wav/demo_2_steps/x_denormlize_2_to_wavform.wav"
sf.write(wav_path,wav_raw,22050, "PCM_24")