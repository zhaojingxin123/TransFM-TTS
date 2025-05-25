import os
import commons
from text import cleaners
import argparse
import datetime as dt
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN

from matcha.models.matcha_tts_noemo import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir, intersperse

'''这个是无emo向量的'''
# 导入函数
# from folder1.script1 import function_name
# from vits_pinyin import VITS_PinYin
MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}

MULTISPEAKER_MODEL = {
    "matcha_vctk": {"vocoder": "hifigan_univ_v1", "speaking_rate": 0.85, "spk": 0, "spk_range": (0, 174)}
}

SINGLESPEAKER_MODEL = {"matcha_ljspeech": {"vocoder": "hifigan_T2_v1", "speaking_rate": 0.95, "spk": None}}

# 
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

# --------match,lj的-------------#
def process_text(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")

    
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]

    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

# 处理文本，这里需要可以改成aishell的,调用太麻烦，全都堆在了这里
#--------------aishell的---------------------#


symbols_TW=[ "_",
    "\uff0c", "\u3002","\uff01","\uff1f","\u2014","\u2026","\u3105","\u3106","\u3107",
    "\u3108","\u3109", "\u310a", "\u310b","\u310c","\u310d","\u310e","\u310f","\u3110",
    "\u3111","\u3112","\u3113", "\u3114","\u3115","\u3116", "\u3117","\u3118","\u3119",
    "\u311a","\u311b", "\u311c","\u311d","\u311e","\u311f","\u3120","\u3121","\u3122",
    "\u3123","\u3124","\u3125","\u3126","\u3127","\u3128","\u3129","\u02c9","\u02ca", 
    "\u02c7", "\u02cb","\u02d9",
    " "
  ]

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

def text_to_sequenceTW(text, symbols, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
#   print(symbols)
#   print("------------symbols的长度-------------",len(symbols))
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  clean_text = _clean_text(text, cleaner_names)
  # print(clean_text)
  # print(f" length:{len(clean_text)}")
  for symbol in clean_text:
    if symbol not in symbol_to_id.keys():
      continue
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  # print(f"length:{len(sequence)}")
  return sequence,clean_text

def process_text_aishell_TW(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    text_norm,clean_text = text_to_sequenceTW(text, symbols_TW,["chinese_cleaners"])
    # print(text_norm)
    # print(type(text_norm))
    # print(f'hps.symbols:{symbols_TW}')
    # print("-----------*-*-*-*-*-----------------",len(symbols_TW))
    if True:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm= torch.tensor(text_norm, dtype=torch.long,device=device)[None]
    x=text_norm
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)


    return {"x_orig": text, "x": x, "x_lengths":x_lengths, "x_phones": clean_text}





def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    # save_dir = get_user_data_dir()
    # if not hasattr(args, "checkpoint_path") and args.checkpoint_path is None:
    #     model_path = args.checkpoint_path
    # else:
    #     model_path = save_dir / f"{args.model}.ckpt"
    #     assert_model_downloaded(model_path, MATCHA_URLS[args.model])

    # vocoder_path = save_dir / f"{args.vocoder}"
    # assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    # 直接改成了模型位置
    model_path=args.checkpoint_path
    # model_path='/home/zjx/Matcha-TTS/logs/train/aishell3_emo/runs/2024-08-26_00-22-01/checkpoints/checkpoint_epoch=169.ckpt'
    vocoder_path='/home/zjx/Matcha-TTS/checkpoint/hifigan_univ_v1'
    return {"matcha": model_path, "vocoder": vocoder_path}


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


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    # print(model)
    _ = model.eval()

    print(f"[+] {model_name} loaded!")
    return model


def to_waveform(mel, vocoder, denoiser=None):
    # print("mel.device----",mel.device)
   
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    # 将folder字符串转换为Path对象
    folder = Path(folder)
    # 创建文件夹，如果不存在则创建，parents=True表示创建所有必要的父文件夹，exist_ok=True表示如果文件夹已存在则不会报错
    folder.mkdir(exist_ok=True, parents=True)
    
    # 生成保存图像的路径，文件名为filename_emo_name.png
    plot_path=folder / f"{filename}.png"
    # 打印图像保存路径
    print(plot_path)

    # 将output中的mel谱转换为numpy数组，并保存为图像文件
    plot_spectrogram_to_numpy(np.array(output["mel"].squeeze().float().cpu()), plot_path)
    # 
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")
    
    return folder.resolve() / f"{filename}.wav"


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        # When using pretrained models
        if args.model in SINGLESPEAKER_MODEL:
            args = validate_args_for_single_speaker_model(args)

        if args.model in MULTISPEAKER_MODEL:
            args = validate_args_for_multispeaker_model(args)
    else:
        # When using a custom model
        if args.vocoder != "hifigan_univ_v1":
            warn_ = "[-] Using custom model checkpoint! I would suggest passing --vocoder hifigan_univ_v1, unless the custom model is trained on LJ Speech."
            warnings.warn(warn_, UserWarning)
        if args.speaking_rate is None:
            args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args


def validate_args_for_multispeaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != MULTISPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {MULTISPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = MULTISPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = MULTISPEAKER_MODEL[args.model]["speaking_rate"]

    spk_range = MULTISPEAKER_MODEL[args.model]["spk_range"]
    if args.spk is not None:
        assert (
            args.spk >= spk_range[0] and args.spk <= spk_range[-1]
        ), f"Speaker ID must be between {spk_range} for this model."
    else:
        available_spk_id = MULTISPEAKER_MODEL[args.model]["spk"]
        warn_ = f"[!] Speaker ID not provided! Using speaker ID {available_spk_id}"
        warnings.warn(warn_, UserWarning)
        args.spk = available_spk_id

    return args


def validate_args_for_single_speaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != SINGLESPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {SINGLESPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = SINGLESPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = SINGLESPEAKER_MODEL[args.model]["speaking_rate"]

    if args.spk != SINGLESPEAKER_MODEL[args.model]["spk"]:
        warn_ = f"[-] Ignoring speaker id {args.spk} for {args.model}"
        warnings.warn(warn_, UserWarning)
        args.spk = SINGLESPEAKER_MODEL[args.model]["spk"]

    return args


@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="matcha_ljspeech",
        help="Model to use",
        choices=MATCHA_URLS.keys(),
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the custom model checkpoint",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default='hifigan_univ_v1',
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
        choices=VOCODER_URLS.keys(),
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    # parser.add_argument("--emo_path", type=str, default=None, help="Emotion to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of ODE steps  (default: 10)")
    
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )

    args = parser.parse_args()

    args = validate_args(args)
    print(f"args.spk::::{args.spk}")

    device = get_device(args)
    print_config(args)
    # 这里是判断有没有原始参数的的并不影响
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device)
    # 
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)

    texts = get_texts(args)
    # args.spk=11
    # print(f"args.spk::::{args.spk}")
    spk = torch.tensor([args.spk], device=device, dtype=torch.long) if args.spk is not None else None

    # --------------------------------------------------------------
    # emo_path=args.emo_path
    
    # emo_path = "/home/zjx/Matcha-TTS/vits_chinese-bert_vits_aishell/train/wav16k/SSB0535/SSB05350001.wav.emo.pt"


    # emo_path = "/home/zjx/Matcha-TTS/vits_chinese-bert_vits_aishell/train/wav16k/SSB0057/SSB00570002.wav.emo.pt"
    
    # emo.to(device)

    if len(texts) == 1 or not args.batched:
        # ---
        unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
    else:
        batched_synthesis(args, device, model, vocoder, denoiser, texts, spk)


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x = []
    x_lengths = []

    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}


def batched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    processed_text = [process_text_aishell_TW(i, text, "cpu") for i, text in enumerate(texts)]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        b = batch["x"].shape[0]
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk.expand(b) if spk is not None else spk,
            length_scale=args.speaking_rate,
        )

        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder,)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")

# 不批量的合成
def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        # 这里换成vits的方式
        text_processed = process_text_aishell_TW(i, text, device)
        # 
        # emo=torch.load(emo_path)
        # emo=emo.unsqueeze(0)
        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            # emo,
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
        )
        print('***************n_timesteps:',args.steps)
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        # emo_name=emo_path.split('/')[-1].split('.')[0]
        # print(f"emo_name:{emo_name}")
        # 这里是单条合成的mel保存，保存mel.png,wav文件
        location = save_to_folder(base_name, output, args.output_folder)
        print(f"[+] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    
    return device


if __name__ == "__main__":
    cli()
