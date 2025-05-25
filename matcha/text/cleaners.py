""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
# 文本字符串处理的库
import re
from pypinyin import lazy_pinyin, BOPOMOFO
import jieba
import phonemizer
from unidecode import unidecode
import cn2an

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'),
    ('b', 'ㄅㄧˋ'),
    ('c', 'ㄙㄧˉ'),
    ('d', 'ㄉㄧˋ'),
    ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'),
    ('g', 'ㄐㄧˋ'),
    ('h', 'ㄝˇㄑㄩˋ'),
    ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'),
    ('k', 'ㄎㄟˋ'),
    ('l', 'ㄝˊㄛˋ'),
    ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'),
    ('o', 'ㄡˉ'),
    ('p', 'ㄆㄧˉ'),
    ('q', 'ㄎㄧㄡˉ'),
    ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'),
    ('t', 'ㄊㄧˋ'),
    ('u', 'ㄧㄡˉ'),
    ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
    ('x', 'ㄝˉㄎㄨˋㄙˋ'),
    ('y', 'ㄨㄞˋ'),
    ('z', 'ㄗㄟˋ')
]]


# 这里这里是音素化的工具库，我自己找的
# https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)


# 这个方案不行，多音字不太好解决好像
global_phonemizer_chinese = phonemizer.backend.EspeakBackend(
    language="cmn",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# 扩充缩写
def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()

# 用匹配一个到多个空格
def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
# 非中文的选择这个
# 在这里转换为中文试一下。
def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number,cn2an.an2cn(number), 1)
    return text

def chinese_to_bopomofo(text):
    text = text.replace('、', '，').replace('；', '，').replace('：', '，')
    # 使用jieba分词转换
    words = jieba.lcut(text, cut_all=False)
    text = ''
    for word in words:
        # 这里的 BOPOMOFO = STYLE_BOPOMOFO = Style.BOPOMOFO
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not re.search('[\u4e00-\u9fff]', word):
            text += word
            continue
        for i in range(len(bopomofos)):

            bopomofos[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopomofos[i])
        if text != '':
            text += ' '
        text += ''.join(bopomofos)

    return text
def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text

# 这里参考vits的
def chinese_cleaners(text):
    '''Pipeline for Chinese text'''
    # 去掉标记符
    text = text.replace("[ZH]", "")
    # 数字转换为中文
    text = number_to_chinese(text)
    # 中文转换为音标
    text = chinese_to_bopomofo(text)
    # 
    text = latin_to_bopomofo(text)
    text = re.sub(r'([ˉˊˇˋ˙])$', r'\1。', text)
    return text





# 把这里改成中文应该就可以训练了
def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    # 文本进行清洗
    # 估计问题是出在这里了
    # 对文本变成音素,所以最后返回的是音素
    # 这里为什么是单线程的，不行的话可以预处理好的
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    # 去除空格
    print('获取音素')
    phonemes = collapse_whitespace(phonemes)
    # 最后返回音素
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes
