import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as torchaudio
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


#--------------#
def strings_to_numbers(string_list):
    # 将字符串列表转换为整数列表
    number_list = [int(s) for s in string_list]
    # 将整数列表转换为 NumPy 数组
    array = np.array(number_list)
    
    return array






filepaths_and_text=[]
def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
      
        # 之前这里只有两个，现在aishell3有好多
        # filepaths_and_text = [line.strip().split(split_char) for line in f]
      for line in f:
          aa=line.strip().split(split_char) 
        # 从txt里面读出来的'[1,2,1,5]'给爷整笑了
        # {speaker_map[speaker]}|{wave_file}|{spec_file}|{char_embeds_path}|{phone_items_str}|{text}|{phone_ID}
        # filepaths_and_text=[wave_file,text,spec_file,phone_list,{speaker_map[speaker]},phone_id]
        #   print(aa)
        #   print(aa[-1])
        #   bb=aa[-1][1:-1]
        #   print(bb)
        #   bb.split(',')
        #   phone_ID=[int(num) for num in bb.split(',')]
        #   print(type(phone_ID[1]))
          filepaths_and_text.append([aa[1],aa[5],aa[2],aa[4],aa[0],aa[6]])
        #   print([aa[1],aa[5],aa[2],phone_ID])
        
    return filepaths_and_text

filepaths_and_text=[]
def parse_filelist4esd(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
      
        # 之前这里只有两个，现在aishell3有好多
        # filepaths_and_text = [line.strip().split(split_char) for line in f]
      for line in f:
          aa=line.strip().split(split_char) 
        # 从txt里面读出来的'[1,2,1,5]'给爷整笑了
        # {speaker_map[speaker]}|{wave_file}|{spec_file}|{char_embeds_path}|{phone_items_str}|{text}|{phone_ID}
        # filepaths_and_text=[wave_file,text,spec_file,phone_list,{speaker_map[speaker]},phone_id]
        #   print(aa)
        #   print(aa[-1])
        #   bb=aa[-1][1:-1]
        #   print(bb)
        #   bb.split(',')
        #   phone_ID=[int(num) for num in bb.split(',')]
        #   print(type(phone_ID[1]))
          filepaths_and_text.append([aa[1],aa[5],aa[2],aa[4],aa[0],aa[6]])
        #   print([aa[1],aa[5],aa[2],phone_ID])
        
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
    ):
        # print(f'这个文件路径读不进来吗？filelist_path：：：{filelist_path}')
        # self.filepaths_and_text = parse_filelist(filelist_path)
        self.filepaths_and_text = parse_filelist4esd(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        # print(f'********************filepath_and_text{filepath_and_text}')
        # 这个也不对了，因为没有添加，多说话人的id
        # filepaths_and_text=[wave_file,text,spec_file,phone_ID,{speaker_map[speaker]}]
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[4]),
                filepath_and_text[1],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None
        # 得到mel，文本
        # 这两部分可以直接加载了
        # 这里的text对于match来说是phoneID
        #-------------------aishell3这里是不用处理了可以直接从处理好的文件里面读------------------------------#  
          
        text, cleaned_text = self.get_text(filepath_and_text, add_blank=self.add_blank)
        mel = self.get_mel(filepath_and_text)

        # 加载一个emo文件此处还差把我路径文件添加上，emo.pt
        emo=torch.from_numpy(np.load(filepath_and_text[-1]))
        ##---------------------------------------------------------------##

        #-------------------------match-------------------------------------#
        # text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        # mel = self.get_mel(filepath)
        ##---------------------------------------------------------------##

        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {"x": text, "y": mel, "spk": spk, "filepath": filepath, "x_text": cleaned_text, "durations": durations,"emo":emo}

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs
    # 这里可以吧mel存起来，每次直接读
    
    def get_mel(self, filepath):
        aa=filepath
        # print(aa)
        mel=torch.load(filepath[2])
        # norm
        # print('读取处理完的mel')
        # mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel
    
    # gaichengaishell3，之前传入的是text
    def get_text(self, filepath_and_text, add_blank=True):
        #--------------------------针对match-------------------------------------#
        # text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        ##---------------------------------------------------------------##
        # 加载tensor值
        text_norm= torch.load(filepath_and_text[5])
        # print(type(text_norm[0]))
        # print(text_norm)
        # bb=text_norm[1:-1]
        # print(bb)
        # bb.split(',')
        # text_norm=[int(num) for num in bb.split(',')]
        # print(type(text_norm[1]))
        # # 成字符串了，不改了，自己搞回去吧
        # text_norm = strings_to_numbers(text_norm)
        # print(text_norm)
        # text_norm=torch.from_numpy(result_array)
        cleaned_text=filepath_and_text[1]

        # 台湾拼音版本的不用了，直接加载就行了
        # 对已经转换为phoneID的list进行0插值

        # if self.add_blank:
        #     # 对序列进行 0插值
        #     text_norm = intersperse(text_norm, 0)
        # text_norm = torch.IntTensor(text_norm)
        # print('读取处理完的文本IDseq和文本')
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        # 返回的值
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)

# Collate_fn
class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks
    # Collate函数
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)

        x_max_length = max([item["x"].shape[-1] for item in batch])

        n_feats = batch[0]["y"].shape[-2]
        # 组成统一大小的batch
        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)

        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []

        #把一个batch的数据整理好 
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            # 把x，y的真实长度存起来
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            # 把数据放入容器
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_

            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]
        # 把想，y的长度变成tensor
        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None
        # print('collate_fn,返回Dataloader的东西')


        emo=item["emo"]


        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
            "emo": emo
        }
