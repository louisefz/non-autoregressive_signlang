# -*- coding: utf-8 -*-


# !pip install wandb
# # !pip install sacrebleu
# !pip install dill

"""## Import Libraries"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn


import torch.nn as nn
from copy import deepcopy
import math
import numpy as np

from torch.autograd import Variable
from torch import Tensor



from torch.cuda import device_count

import time
# import wandb
import string



from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
# from sacrebleu import corpus_bleu



from torch.utils.data import Dataset, DataLoader

from datetime import datetime
import torch
import copy
import collections
# import dill

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

from google.colab import drive
drive.mount('/content/gdrive')

"""# Data and Model Config

## Model config
"""

BATCH_MULTIPLIER = 1.
SINGLE_GPU_BATCH_SIZE = 16
BATCH_SIZE = SINGLE_GPU_BATCH_SIZE * device_count()
MAX_DECODER_RATIO = 2.
print(device_count())

config = {'num_layers': 6,
          'attn_heads': 8,
          'ff_dim': 2048,
          'model_dim': 512,
          'single_gpu_real_batch_size': SINGLE_GPU_BATCH_SIZE,
          'batch_size': BATCH_SIZE,
          'batch_multiplier': BATCH_MULTIPLIER,
          'effective_batch_size': BATCH_SIZE * BATCH_MULTIPLIER,
          'val_batch_size': SINGLE_GPU_BATCH_SIZE * BATCH_MULTIPLIER // (MAX_DECODER_RATIO * 2),
          'max_decode_iter': 10,
          'max_len': 150,
          'min_freq': 1,
          'warmup_init_lr': 1e-07,
          'warmup_end_lr': 0.0005,
          'min_lr': 1e-09,
          'warmup': 10000,
          'dropout': 0.0,
          'input_dropout': 0.3,
          'weight_decay': 0.01,
          'epsilon': 1e-9,
          'max_step': 3e5,
          'beta_1': 0.9,
          'beta_2': 0.98,
          'decoder_insertion_ratio': MAX_DECODER_RATIO / 3,
          'decoder_length_ratio': MAX_DECODER_RATIO
          }

"""## Data preprocessing

### data loading
"""

# class dataprocess4signgloss:
#     def __init__(self, data_file_gloss,data_file_standard, task="mask_filling"):  # task = "mask_filling" or "translation"
#         self.data_file_gloss = data_file_gloss
#         self.data_file_standard = data_file_standard
#         self.task = task

#     def df(self):


#         dfs_Sign_gloss = []
#         dfs_Standard_text = []

#         dfs_Sign_gloss_token = []
#         dfs_Standard_text_token = []



#         with open(self.data_file_gloss,encoding='utf-8') as file_gloss:
#             text_list_gloss = file_gloss.readlines()

#         with open(self.data_file_standard,encoding='utf-8') as file_standard:
#             text_list_standard = file_standard.readlines()

#         for sent_gloss, sent_standard in zip(text_list_gloss,text_list_standard):
#             sent_lower_gloss = sent_gloss.lower().replace('\n','') #change uppercase to lowercase
#             translator = str.maketrans('', '', string.punctuation)
#             sent_gloss_clean = sent_lower_gloss.translate(translator)
#             # if self.task == "mask_filling":
#             #     sentence_gloss = sent_gloss_clean.replace(" "," [MASK] ")
#             # else:
#             sentence_gloss = sent_gloss_clean.replace(" ", "")
#             sentence_gloss_token = sent_gloss_clean
#             sent_lower_standard = sent_standard.lower().replace('\n','')
#             sentence_standard = sent_lower_standard.translate(translator).replace(' ', '')
#             sentence_standard_token = sent_lower_standard.translate(translator)




#             dfs_Sign_gloss.append(sentence_gloss)
#             dfs_Standard_text.append(sentence_standard)
#             dfs_Sign_gloss_token.append(sentence_gloss_token)
#             dfs_Standard_text_token.append(sentence_standard_token)
#         data = {
#             'Sign_gloss': dfs_Sign_gloss,
#             'Standard_text':dfs_Standard_text
#         }
#         data_token = {
#             'Sign_gloss_token': dfs_Sign_gloss_token,
#             'Standard_text_token':dfs_Standard_text_token
#         }
#         df_data = pd.DataFrame(data)
#         df_data_token = pd.DataFrame(data_token)


#         return df_data, df_data_token


class dataprocess4signgloss:
    def __init__(self, data_file):
        self.data_file = data_file


    def df(self):


        dfs_Sign_gloss = []
        dfs_Sign_gloss_reordered = []
        dfs_Standard_text = []
        is_real = []



        with open(self.data_file,encoding='utf-8') as file_gloss:
            lines = file_gloss.readlines()



        for i in range(0, len(lines), 4):
            gloss = lines[i].strip()
            reordered_gloss = lines[i+1].strip()
            text = lines[i+2].strip()
            label = lines[i+3].strip()




            dfs_Sign_gloss.append(gloss)
            dfs_Sign_gloss_reordered.append(reordered_gloss)
            dfs_Standard_text.append(text)
            is_real.append(label)
        data = {
            'Sign_gloss': dfs_Sign_gloss,
            'Sign_gloss_reordered':dfs_Sign_gloss_reordered,
            'Standard_text':dfs_Standard_text,
            'is_real':is_real
        }

        df_data = pd.DataFrame(data)


        return df_data

# gloss_file = "/content/gdrive/MyDrive/Colab Notebooks/sign_language/1_dataset/ZH_CLS/zh_gloss.txt"
# standard_file = "/content/gdrive/MyDrive/Colab Notebooks/sign_language/1_dataset/ZH_CLS/zh_standard.txt"

# data = dataprocess4signgloss(gloss_file,standard_file, task="mask_filling")



gloss_file = "/content/gdrive/MyDrive/Colab Notebooks/sign_language/0_data_augmentation/monolingual_dataset/all_english_sentences_LevTpointer_less_50_len.txt"

data = dataprocess4signgloss(gloss_file)

# print(data.df())

# data_df,data_token_df= data.df()


data_df = data.df()
print(data_df.head(5))

# X = data_df[['Sign_gloss']]
# y = data_df['Standard_text']

# X_token = data_token_df[['Sign_gloss_token']]
# y_token = data_token_df['Standard_text_token']

# X_train, X_test_val, y_train, y_test_val = train_test_split(X,y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
# X_train_token, X_test_val_token, y_train_token, y_test_val_token = train_test_split(X_token,y_token, test_size=0.3, random_state=42)
# X_val_token, X_test_token, y_val_token, y_test_token = train_test_split(X_test_val_token, y_test_val_token, test_size=0.5, random_state=42)


# print(X_train)




# X = data_df[['Sign_gloss']]
# y = data_df['Standard_text']

# X_train, X_test_val, y_train, y_test_val = train_test_split(X,y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

X = data_df[['Sign_gloss', 'is_real']]  # 包括 'is_real' 列以便于分割
y = data_df[['Sign_gloss_reordered','Standard_text']]
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=data_df['is_real'])
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=X_test_val['is_real'])


y_train_tgt1 = y_train[['Sign_gloss_reordered']].copy()
y_train_tgt2 = y_train[['Standard_text']].copy()

y_val_tgt1 = y_val[['Sign_gloss_reordered']].copy()
y_val_tgt2 = y_val[['Standard_text']].copy()

print(y_train_tgt1)
print(y_train_tgt2)

# print(y_train.to_string(index=False))
is_real_counts = X_train['is_real'].value_counts()
# print(is_real_counts)
is_real_counts1 = X_val['is_real'].value_counts()
# print(is_real_counts1)
is_real_counts2 = X_test['is_real'].value_counts()
# print(is_real_counts2)

X_train = X_train.drop(columns=['is_real'])
X_val = X_val.drop(columns=['is_real'])


# print(X_train.head(5))
# print(X_val.head(5))

# 对于测试集，要对合成数据和真实数据分别进行test
X_test_real = X_test[X_test['is_real'] == '1']
# print(X_test_real.head(5))
X_test_real = X_test_real.drop(columns=['is_real'])

X_test_fake = X_test[X_test['is_real'] == '0']
# print(X_test_fake.head(5))
X_test_fake = X_test_fake.drop(columns=['is_real'])

X_test_all = X_test.drop(columns=['is_real'])

y_test_tgt1 = y_test[['Sign_gloss_reordered']].copy()
y_test_tgt2 = y_test[['Standard_text']].copy()

y_test_real = y_test.loc[X_test_real.index]
y_test_fake = y_test.loc[X_test_fake.index]

y_test_real_tgt1 = y_test_real[['Sign_gloss_reordered']].copy()
y_test_real_tgt2 = y_test_real[['Standard_text']].copy()

y_test_fake_tgt1 = y_test_fake[['Sign_gloss_reordered']].copy()
y_test_fake_tgt2 = y_test_fake[['Standard_text']].copy()

# # print(X_test_real.head(5))
# print(y_test_real.head(5))
# # print(X_test_fake.head(5))
# print(y_test_fake.head(5))
# # print(X_test_fake.head(5))
# # print(X_test_real_all.head(5))

# print(y_test_fake_tgt1)
# print(y_test_fake_tgt2)

"""###batch setting_defaut"""

max_src_in_batch = config['batch_size']
max_tgt_in_batch = config['batch_size']
print(max_src_in_batch)
print(max_tgt_in_batch)


def batch_size_fn(new, count, size_so_far):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch
    global max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def rebatch_and_noise(batch, pad: int, bos: int, eos: int):
    """
    Fix order in torchtext to match ours
    """
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    # src, trg = batch['input_ids'], batch['attention_mask']
    return BatchWithNoise(src=src, trg=trg, pad=pad, bos=bos, eos=eos)


class BatchWithNoise(object):
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, pad: int, bos: int, eos: int, trg=None):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # self.noised_trg = inject_noise(trg, pad=pad, bos=bos, eos=eos)
            # self.noised_trg_mask = self.make_std_mask(self.noised_trg, pad)
            self.ntokens = (trg[:, 1:] != pad).data.sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

"""### batch setting_dataloader

我不太确定 source_mask和attention_mask是不是一个东西
"""

class TranslationDataset(Dataset):
    def __init__(self, X, y1,y2, tokenizer, max_length=255):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # 从数据集中获取源文本和目标文本
        source_text = str(self.X.iloc[idx][0])#.split("    ")[1].split("\n")[0]
        target_text_reordered = str(self.y1.iloc[idx][0])
        target_text = str(self.y2.iloc[idx][0])

        # 对源文本和目标文本进行Tokenization
        # 注意: 我们为源文本和目标文本使用不同的处理方法，尤其是当目标文本被用作 labels 时
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 目标文本用于计算损失，不需要attention_mask

        target_reordered_encoding = self.tokenizer(
            target_text_reordered,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        #noise tokens
        ntokens = (target_encoding['input_ids'][1:] != 1).sum().item()

        return {
        'input_ids': source_encoding['input_ids'].squeeze(0),
        'attention_mask': source_encoding['attention_mask'].squeeze(0),
        'labels_tgt1': target_reordered_encoding['input_ids'].squeeze(0),
        'labels_tgt2': target_encoding['input_ids'].squeeze(0),  # 正确设置labels为目标文本的input_ids
        'source_text': source_text,
        'target_text_reordered': target_text_reordered,
        'target_text': target_text,
        'ntokens': ntokens,
        'seq_len': 255
        }

tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', special_tokens=True)
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "en_XX"

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", special_tokens=True)
# tokenizer.src_lang = "en_XX"
# tokenizer.tgt_lang = "en_XX"

len(tokenizer.vocab)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

# train_dataset = TranslationDataset(X_train, y_train, tokenizer)
# val_dataset = TranslationDataset(X_val, y_val, tokenizer)
# test_dataset = TranslationDataset(X_test, y_test, tokenizer)


# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)




# 准备数据集
train_dataset = TranslationDataset(X_train, y_train_tgt1, y_train_tgt2, tokenizer)
val_dataset = TranslationDataset(X_val, y_val_tgt1, y_val_tgt2, tokenizer)
test_dataset_all = TranslationDataset(X_test_all, y_test_tgt1, y_test_tgt2, tokenizer)
test_dataset_real = TranslationDataset(X_test_real, y_test_real_tgt1, y_test_real_tgt2, tokenizer)
test_dataset_fake = TranslationDataset(X_test_fake, y_test_fake_tgt1, y_test_fake_tgt2, tokenizer)


# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader_all_tgt1 = DataLoader(test_dataset_all, batch_size=16, shuffle=False)
test_loader_real_tgt1 = DataLoader(test_dataset_real, batch_size=16, shuffle=False)
test_loader_fake_tgt1 = DataLoader(test_dataset_fake, batch_size=16, shuffle=False)

"""# Utilities

### tokenizer

把tensor转化成token
"""

def load_models(example_model, paths):
    models = []
    for path in paths:
        model = copy.deepcopy(example_model)
        model.load_state_dict(torch.load(path))
        models.append(copy.deepcopy(model))
    return models


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.parameters() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def save_model(model, optimizer, loss, src_field, tgt_field, updates, epoch, prefix=''):
    if prefix != '':
        prefix += '_'
    current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
    file_name = prefix + 'en-de__' + current_date + '.pt'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'updates': updates
    }, file_name)
    torch.save(src_field, f'SRC_{len(src_field.vocab.itos)}.pt', pickle_module=dill)
    torch.save(tgt_field, f'TGT_{len(tgt_field.vocab.itos)}.pt', pickle_module=dill)
    print(f'Model is saved as {file_name}')


def bpe_to_words(sentence):
    new_sentence = []
    for i in range(len(sentence)):
        word = sentence[i]
        if word[-2:] == '@@' and i != len(sentence) - 1:
            sentence[i + 1] = word[:-2] + sentence[i + 1]
        else:
            new_sentence.append(word)
    return new_sentence

# 这里要改一下，用autokenizer进行解码
def vector_to_sentence(vector: torch.Tensor, field, eos_word: str, start_from=1, change_encoding=False):

    sentence = []
    for l in range(start_from, vector.size(0)):
        word = field.vocab.itos[vector[l]]
        if word == eos_word:
            break
        sentence.append(word)

    sentence = ' '.join(bpe_to_words(sentence))
    if change_encoding:
        # fixing encoding
        sentence = sentence.encode('utf-8').decode('latin-1')
    return sentence


def get_src_mask(src, BLANK_WORD_IDX):
    return (src != BLANK_WORD_IDX).unsqueeze(-2)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    return averaged_params

"""### positional label prediction

预测不同位置的label: keep, delete, insert
"""

def suggested_ed2_path(xs: list, ys: list, terminal_symbol: int):
    seq = []
    for i, _ in enumerate(xs):
        distance = edit_distance2_with_dp(xs[i], ys[i])
        seq.append(edit_distance2_backtracking(
            distance, xs[i], ys[i], terminal_symbol))

    return seq


def edit_distance2_with_dp(x: list, y: list):
    l_x = len(x)
    l_y = len(y)
    distance = [[0 for _ in range(l_y + 1)] for _ in range(l_x + 1)]

    for i in range(l_x + 1):
        distance[i][0] = i

    for j in range(l_y + 1):
        distance[0][j] = j

    for i in range(1, l_x + 1):
        for j in range(1, l_y + 1):
            distance[i][j] = min(
                min(distance[i - 1][j], distance[i][j - 1]) + 1,
                distance[i - 1][j - 1] + 2 * (0 if x[i - 1] == y[j - 1] else 1)
            )
    return distance


def edit_distance2_backtracking(distance, x: list, y: list, terminal_symbol: int):
    l_x = len(x)
    edit_seqs = [[] for _ in range(l_x + 2)]
    seq = []

    if l_x == 0:
        edit_seqs[0] = y
        return edit_seqs

    i = len(distance) - 1
    j = len(distance[0]) - 1

    while i >= 0 and j >= 0:
        if i == 0 and j == 0:
            break

        if j > 0 and distance[i][j - 1] < distance[i][j]:
            seq.append(1)  # insert
            seq.append(y[j - 1])
            j -= 1
        elif i > 0 and distance[i - 1][j] < distance[i][j]:
            seq.append(2)  # delete
            seq.append(x[i - 1])
            i -= 1
        else:
            seq.append(3)  # keep
            seq.append(x[i - 1])
            i -= 1
            j -= 1

    prev_op = 0
    s = 0
    l_s = len(seq)

    for k in range(l_s // 2):
        op = seq[l_s - 2 * k - 2]
        word = seq[l_s - 2 * k - 1]
        if prev_op != 1:
            s += 1
        if op == 1:  # insert
            edit_seqs[s - 1].append(word)
        elif op == 2:  # delete
            edit_seqs[l_x + 1].append(1)
        else:
            edit_seqs[l_x + 1].append(0)

        prev_op = op

    for _, edit_seq in enumerate(edit_seqs):
        if len(edit_seq) == 0:
            edit_seq.append(terminal_symbol)

    return edit_seqs


if __name__ == "__main__":
    padding_idx = 100

    x_s = torch.tensor([[1, 2, 3, 4, 5]])
    y_s = torch.tensor([[100, 100, 4, 3, 6, 5]])

    x_s = [[t for t in s if t != padding_idx] for i, s in enumerate(x_s.tolist())]
    y_s = [[t for t in s if t != padding_idx] for i, s in enumerate(y_s.tolist())]

    print(f'{x_s} => {y_s}')
    print(suggested_ed2_path(x_s, y_s, padding_idx))

"""### action: insert,delete, keep"""

# based on fairseq.libnat


def _get_ins_targets(pred: Tensor, target: Tensor, padding_idx: int, unk_idx: int):
    """
    :param pred: Tensor
    :param target: Tensor
    :param padding_idx: long
    :param unk_idx: long
    :return: word_pred_input, word_pred_tgt_masks, ins_targets
    """


    in_seq_len = pred.size(1)
    # print('pred_shape:',pred.shape)
    out_seq_len = target.size(1)
    # print('_target_shape:',target.shape)

    with torch.cuda.device_of(pred):
        # removing padding
        pred_list = [[t for t in s if t != padding_idx] for i, s in enumerate(pred.tolist())]
        target_list = [[t for t in s if t != padding_idx] for i, s in enumerate(target.tolist())]

        full_labels = suggested_ed2_path(pred_list, target_list, padding_idx)

        # get insertion target with number of insertions eg. [0, 2, 1, 0, 2, 0]
        insertion_tgts = [[len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels]

        # generate labels
        word_pred_tgt_masks = []
        for insertion_tgt in insertion_tgts:
            word_gen_mask = []
            # get mask for word generation, eg: [0, 1, 1, 0, 1, 0, 0, 1, 1]
            for beam_size in insertion_tgt[1:-1]:  # HACK 1:-1
                word_gen_mask += [0] + [1 for _ in range(beam_size)]

            # add padding
            word_pred_tgt_masks.append(word_gen_mask + [0 for _ in range(out_seq_len - len(word_gen_mask))])

        ins_targets = [
            insertion_tgt[1:-1] +
            [0 for _ in range(in_seq_len - 1 - len(insertion_tgt[1:-1]))]
            for insertion_tgt in insertion_tgts
        ]

        # transform to tensor


        # word_pred_tgt_masks = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, ..., 0]
        if len(word_pred_tgt_masks) > 250:
            word_pred_tgt_masks = word_pred_tgt_masks[:250]
        word_pred_tgt_masks = torch.tensor(word_pred_tgt_masks, device=target.device).bool()


        # ins_targets = [0, 2, 1, 0, 2, 0, 0, 0, ..., 0]
        ins_targets = torch.tensor(ins_targets, device=pred.device)

        # word_pred_tgt = [0, <unk>, <unk>, 0, <unk>, 0, 0, <unk>, <unk>, 0, 0, ..., 0]
        word_pred_input = target.masked_fill(word_pred_tgt_masks, unk_idx)
        # print('ins_targets2:', ins_targets.shape)
        # print('word_pred_input:', word_pred_input.shape)
        # print('word_pred_tgt_masks2:', word_pred_tgt_masks.shape)



    return word_pred_input, word_pred_tgt_masks, ins_targets



def _get_del_targets(prediction, target, padding_idx):
    out_seq_len = target.size(1)

    with torch.cuda.device_of(prediction):
        prediction_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(prediction.tolist())
        ]
        target_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(target.tolist())
        ]

        # get labels in form of [insert1, insert2, ..., insertn, [del1, del2, ..., deln]]
        full_labels = suggested_ed2_path(
            prediction_list, target_list, padding_idx
        )

        word_del_targets = [b[-1] for b in full_labels]
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # transform to tensor
        word_del_targets = torch.tensor(word_del_targets, device=target.device)
    return word_del_targets


def _get_del_ins_targets(in_tokens, out_tokens, padding_idx):
    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = suggested_ed2_path(in_tokens_list, out_tokens_list, padding_idx)

        # deletion target, eg: [0, 0, 1, 0, 1, 0]
        word_del_targets = [b[-1] for b in full_labels]
        # add padding
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # insertion target with number of words to be inserted, eg [0, 0, 3, 0, 2]
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]
        mask_ins_targets = [
            mask_input[1:-1] +
            [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
            for mask_input in mask_inputs
        ]

        # transform to tensor
        mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
        word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets, mask_ins_targets


def _apply_ins_masks(in_tokens: Tensor, mask_ins_pred: Tensor, pad: int, unk: int, eos: int):
    in_masks = in_tokens.ne(pad)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
            torch.arange(out_max_len, device=out_lengths.device)[None, :]
            < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
            .fill_(pad)
            .masked_fill_(out_masks, unk)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    return out_tokens


def _apply_ins_words(in_tokens: Tensor, word_ins_pred: Tensor, unk: int):
    word_ins_masks = in_tokens.eq(unk)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    return out_tokens


def _apply_del_words(in_tokens: Tensor, word_del_pred: Tensor, pad: int, bos: int,
                     eos: int) -> Tensor:
    # apply deletion to a tensor
    in_masks = in_tokens.ne(pad)
    bos_eos_masks = in_tokens.eq(bos) | in_tokens.eq(eos)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        torch.arange(max_len, device=in_tokens.device)[None, :]
            .expand_as(in_tokens)
            .contiguous()
            .masked_fill_(word_del_pred, max_len)
            .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(
        word_del_pred, pad).gather(1, reordering)

    return out_tokens


# from fairseq model_utils


def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _fill(x: Tensor, mask: Tensor, y: Tensor, padding_idx: int) -> Tensor:
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def inject_noise(target_tokens: Tensor, pad, bos, eos):
    with torch.cuda.device_of(target_tokens):
        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
        target_score.masked_fill_(target_mask, 1)

        # reorder the numbers randomly, with bos and eos at the beginning and paddings at the end
        # ['<bos>', 'asd', 'kek', 'lol', '<bos>', '<pad>', '<pad>', '<pad>'] =>
        # ['<bos>', '<bos>',  'kek', 'lol', 'asd','<pad>', '<pad>', '<pad>']
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(1, keepdim=True)

        # do not delete <bos> and <eos> (we assign 0 score for them)
        # assign a new random length for each line, where: 2 < new_length < original_length
        target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(target_score.size(0), 1).uniform_()).long()
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        # remove tokens after the cutoff
        prev_target_tokens = target_tokens.gather(1, target_rank).masked_fill_(target_cutoff, pad) \
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])

        # remove unnecessary paddings
        prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.ne(pad).sum(1).max()]

    return prev_target_tokens


def initialize_output_tokens(src_tokens: Tensor, bos: int, eos: int):
    initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
    initial_output_tokens[:, 0] = bos
    initial_output_tokens[:, 1] = eos

    return initial_output_tokens


def pad_tensors_in_dim(x_org: Tensor, y_org: Tensor, dim: int, pad: int) -> (Tensor, Tensor):
    x = x_org.detach().clone()
    y = y_org.detach().clone()
    x_shape = [*x.size()]
    y_shape = [*y.size()]

    if y_shape[dim] == x_shape[dim]:
        return x, y
    elif y_shape[dim] > x_shape[dim]:
        pad_shape = x_shape
        pad_shape[dim] = y_shape[dim] - x_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device).fill_(pad)
        padded_x = torch.cat([x, padding_tensor], dim=dim)
        return padded_x, y
    elif y_shape[dim] < x_shape[dim]:
        pad_shape = y_shape
        pad_shape[dim] = x_shape[dim] - y_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=y.dtype, device=y.device).fill_(pad)
        padded_y = torch.cat([y, padding_tensor], dim=dim)
        return x, padded_y


def pad_tensor_to_length(x_org: Tensor, len: int, dim: int, pad: int) -> Tensor:
    x = x_org.detach().clone()
    x_shape = [*x.size()]

    if x_shape[dim] <= len:
        return x
    else:
        pad_shape = x_shape
        pad_shape[dim] = len - x_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device).fill_(pad)
        padded_x = torch.cat([x, padding_tensor], dim=dim)
        return padded_x

"""# Transformer Prepared

## Transformer layers
"""

class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.lookup_table = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        log_softmax can improve numerical performance and gradient optimization compared to softmax.
        https://datascience.stackexchange.com/questions/40714/what-is-the-advantage-of-using-log-softmax-instead-of-softmax
        """
        return F.log_softmax(self.lookup_table(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask = None):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Decoderonly(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, decoder, tgt_embed, generator):
        super(Decoderonly, self).__init__()

        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, encoder_out, tgt: Tensor, src_mask = None, tgt_mask = None):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(encoder_out, src_mask, tgt, tgt_mask)


    def decode(self, memory: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, x_mask = None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, memory: Tensor, src_mask = None, tgt_mask = None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask = None):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, src_mask = None, tgt_mask = None):
        """
        Follow Figure 1 (right) for connections.
        """

        # print(x.size(), memory.size(), src_mask.size(), tgt_mask.size())
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See https://arxiv.org/abs/1607.06450 for details).

    Setting the activations' mean to 0 and standard deviation to 1,
    as in RNNs the inputs tend to either grow or shrink at every step,
    making the gradient vanish or explode.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Tensor):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: Tensor):
        x = x + self.positional_encoding[:, :x.size(1)].clone().detach().requires_grad_(True)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.sqrt_d_model = math.sqrt(self.d_model)

    def forward(self, x: Tensor):
        return self.lookup_table(x) * self.sqrt_d_model

"""## Transformer module"""

def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    # print(query.size(), key.size(), value.size(), mask.size())
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    # pylint: disable=no-member
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

"""## Transformer sublayes"""

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

"""## Transformer optimizer"""

class NoamOpt(object):
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, warmup_init_lr: float, warmup_end_lr: float, warmup_updates: float, min_lr: float, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_updates = warmup_updates
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        self.min_lr = min_lr

        self._rate = warmup_init_lr

        self.warmup_lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step

        if step < self.warmup_updates:
            return self.warmup_init_lr + step * self.warmup_lr_step
        else:
            return max(self.decay_factor * step ** -0.5, self.min_lr)

"""# Model

## Reordering decoder+ Levenshtein decoder

### untilities

### pointer network
"""

def sinkhorn(log_alpha, num_iters=100, epsilon=1e-6):
    """
    Perform Sinkhorn normalization on the input log attention scores.

    Parameters:
    - log_alpha (Tensor): The input log attention scores of shape (batch_size, n, m)
    - num_iters (int): The number of iterations to run the Sinkhorn algorithm
    - epsilon (float): A small value to avoid division by zero

    Returns:
    - log_alpha (Tensor): The normalized attention scores of shape (batch_size, n, m)
    """
    for _ in range(num_iters):
        # Normalize rows
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Normalize columns
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def get_tgt_indices(src, tgt):
    """
    Args:
        src: Tensor of shape (batch_size, seq_len)
        tgt: Tensor of shape (batch_size, seq_len)

    Returns:
        tgt_indices: Tensor of shape (batch_size, seq_len)
    """
    batch_size, seq_len = tgt.shape
    tgt_indices = []

    for i in range(batch_size):
        src_batch = src[i].tolist()
        tgt_batch = tgt[i].tolist()
        tgt_indices_batch = []

        for src_index, token in enumerate(src_batch):
            try:
                index = tgt_batch.index(token)
                tgt_indices_batch.append(index)
            except ValueError:
                # If token not found in tgt, append 254 or handle it accordingly
                tgt_indices_batch.append(src_index)

        tgt_indices.append(tgt_indices_batch)

    # Convert the list of lists to a tensor
    tgt_indices_tensor = torch.tensor(tgt_indices, dtype=torch.long)

    return tgt_indices_tensor

class Attention(nn.Module):
  def __init__(self, hidden_size, # hidden_size = in_features (int): emd_size/input_size  512
               attention_units # attention_units = out_features (int) : size of each output sample
               ):

    super(Attention, self).__init__()
    self.key_layer = nn.Linear(hidden_size, attention_units, bias=False)

    self.value_layer = nn.Linear(hidden_size, attention_units, bias=False)
    self.query_layer = nn.Linear(hidden_size, attention_units, bias=False)

    self.FNN =  nn.Linear(attention_units, 1, bias=False)  # out_feature为1：最后通过一个线性层投影到一个标量分数


  def forward(self,
              encoder_output: torch.Tensor,
              decoder_output: torch.Tensor):
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # decoder_hidden: (BATCH, HIDDEN_SIZE)

    # Add time axis to decoder hidden state
    # in order to make operations compatible with encoder_out
    # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
    # query = self.query_layer(decoder_hidden.unsqueeze(1).to('cuda')).to('cuda')  # (batch_size, 1, hidden_dim
    # keys = self.key_layer(encoder_out.to('cuda')).to('cuda')
    # values = self.value_layer(encoder_out.to('cuda')).to('cuda')
    # dk = query.size(-1)

    # attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(dk)  # decoder hidden对encoder所有的hidden做cross attention，最后得到一个encoder hidden states的relavence distribution

    attention_scores = torch.bmm(decoder_output, encoder_output.transpose(1, 2))

    log_attention_scores = torch.log_softmax(attention_scores, dim=-1)



    normalized_attention_scores = sinkhorn(log_attention_scores) # softmax之后的attention score


    context_vector = torch.matmul(normalized_attention_scores, encoder_output.squeeze(0))  # 它提供了关于输入序列的相关信息，使得解码器在生成下一个输出时能够利用整个输入序列的信息
    # max_positions = torch.argmax(attention_weights.to('cuda'), dim=-1)


    return   context_vector, normalized_attention_scores   #.squeeze(-1)  # 使用 squeeze(-1) 去掉去掉多余的单维度例如(batch_size, 1, seq_len)，使得张量的形状更简洁和符合预期，得到形状为 (batch_size, seq_len)，这使得后续的操作更加简洁和符合预期的维度。





def pointer_network(encoder_out,src, tgt, d_model):


    batch_size, seq_len, feature_dim = encoder_out.size()   # _是tokenizer_embed







    tgt_label = get_tgt_indices(src, tgt).to('cuda')
    # print(tgt_label.shape)
    # print(tgt_label.shape)


    # batch_size, seq_len, hidden_dim = decoder_output.shape
    attention = Attention(d_model, d_model)
    # for t in range(seq_len):

    context_vector, attention_scores = attention(encoder_out,encoder_out)
    argmax_indices = torch.argmax(attention_scores, dim=-1)

    reordered_src = torch.zeros_like(src)

    # 对每个 batch 进行操作
    for b in range(batch_size):
        for i in range(seq_len):
            # 根据 argmax_indices 重新排列 src 和 encoder_outputs
            reordered_src[b, i] = src[b, argmax_indices[b, i]]





    pointer_loss = F.cross_entropy(attention_scores.view(-1, attention_scores.size(1)), tgt_label.reshape(-1))


    return pointer_loss, reordered_src, context_vector

"""### LevT Model"""

class LevenshteinEncodeDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad, bos, eos, unk, criterion):
        super(LevenshteinEncodeDecoder, self).__init__(encoder, decoder, src_embed, tgt_embed, generator)
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.unk = unk
        self.criterion = criterion

        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)




    # 全局视角 初始阶段的forward过程：先insert，然后delete
    def forward(self, src: Tensor, tgt1, tgt: Tensor, d_model):

        # TODO: CHECK OUTPUTS OF ALL TGT-PRED PAIRS!!!
        '''
        从一个较高的层面来看，模型首次被调用时，通常会从一个非常粗略或空白的状态开始，
        首先进行插入操作（生成占位符或初步的词汇），然后根据需要进行删除操作。
        这个过程是指模型整体上如何开始处理一个全新的生成任务，或者在每次迭代之初的大致策略
        '''


        assert tgt is not None, "Forward function only supports training."

        # encoding
        pointer_loss, reordered_src, encoder_out = self.encode(src, tgt1, d_model)



        #这里需要reorder，根据output
        x = reordered_src
        x_mask = BatchWithNoise.make_std_mask(x, pad=self.pad)



        # generate training labels for insertion
        # word_pred_tgt, word_pred_tgt_masks, ins_targets
        word_pred_input, word_pred_tgt_masks, ins_targets = _get_ins_targets(x, tgt, self.pad, self.unk)
        word_pred_tgt_subsequent_masks = BatchWithNoise.make_std_mask(word_pred_input, pad=self.pad)

        ins_targets = ins_targets.clamp(min=0, max=255)  # for safe prediction
        ins_masks = x[:, 1:].ne(self.pad)
        ins_out = self.decoder.forward_mask_ins(encoder_out, self.tgt_embed(x), x_mask)

        word_pred_out = self.decoder.forward_word_ins(encoder_out, self.tgt_embed(word_pred_input),
                                                      word_pred_tgt_subsequent_masks)
        # make online prediction
        word_predictions = F.log_softmax(word_pred_out, dim=-1).max(2)[1]
        word_predictions.masked_scatter_(~word_pred_tgt_masks, tgt[~word_pred_tgt_masks])
        word_predictions_subsequent_mask = BatchWithNoise.make_std_mask(word_predictions, pad=self.pad)

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt, self.pad)
        word_del_out = self.decoder.forward_word_del(encoder_out, self.tgt_embed(word_predictions),
                                                     word_predictions_subsequent_mask)
        word_del_mask = word_predictions.ne(self.pad)

        ins_loss = self.criterion(outputs=ins_out, targets=ins_targets, masks=ins_masks, label_smoothing=0.0)
        word_pred_loss = self.criterion(outputs=word_pred_out, targets=tgt, masks=word_pred_tgt_masks,
                                        label_smoothing=0.1)
        word_del_loss = self.criterion(outputs=word_del_out, targets=word_del_targets,
                                       masks=word_del_mask, label_smoothing=0.01)

        return {
            "ins_loss": ins_loss,
            "word_pred_loss": word_pred_loss,
            "word_del_loss": word_del_loss,
            "loss": ins_loss + word_pred_loss + word_del_loss + pointer_loss,
            "pointer_loss": pointer_loss
        }

    def encode(self, src: Tensor,tgt1, d_model) -> Tensor:
        encoder_out = self.encoder(self.src_embed(src))
        pointer_loss, reordered_src, encoder_out = pointer_network(encoder_out,src, tgt1, d_model)
        return pointer_loss, reordered_src, encoder_out


    def decode(self, encoder_out: Tensor, x: Tensor, encoder_padding_mask: Tensor,
               eos_penalty=0.0, max_ins_ratio=None, max_out_ratio=None) -> Tensor:

        if max_ins_ratio is None:
            max_ins_lens = x.new().fill_(255)
            max_out_lens = x.new().fill_(255)
        else:
            src_lens = encoder_padding_mask.squeeze(1).sum(1)
            max_ins_lens = (src_lens * max_ins_ratio).clamp(min=5).long()
            max_out_lens = (src_lens * max_ins_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = x.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            word_del_out = self.decoder.forward_word_del(
                _skip(encoder_out, can_del_word),
                _skip(encoder_padding_mask, can_del_word),
                self.tgt_embed(_skip(x, can_del_word)),
                _skip(x_mask, can_del_word))

            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens = _apply_del_words(
                x[can_del_word],
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )

            x = _fill(x, can_del_word, _tokens, self.pad)

        # insert placeholders
        can_ins_mask = x.ne(self.pad).sum(1) < max_out_lens
        if can_ins_mask.sum() != 0:
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            mask_ins_out = self.decoder.forward_mask_ins(
                _skip(encoder_out, can_ins_mask),
                _skip(encoder_padding_mask, can_ins_mask),
                self.tgt_embed(_skip(x, can_ins_mask)),
                _skip(x_mask, can_ins_mask))

            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] -= eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            limit_ins_pred = max_ins_lens[can_ins_mask, None].expand_as(mask_ins_pred)
            mask_ins_pred = torch.min(mask_ins_pred, limit_ins_pred)

            _tokens = _apply_ins_masks(
                x[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            x = _fill(x, can_ins_mask, _tokens, self.pad)

        # insert words
        can_ins_word = x.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            word_ins_out = self.decoder.forward_word_ins(
                _skip(encoder_out, can_ins_word),
                _skip(encoder_padding_mask, can_ins_word),
                self.tgt_embed(_skip(x, can_ins_word)),
                _skip(x_mask, can_ins_word))

            word_ins_score = F.log_softmax(word_ins_out, 2)
            word_ins_pred = word_ins_score.max(-1)[1]

            _tokens = _apply_ins_words(
                x[can_ins_word],
                word_ins_pred,
                self.unk,
            )

            x = _fill(x, can_ins_word, _tokens, self.pad)

        # delete some unnecessary paddings
        cut_off = x.ne(self.pad).sum(1).max()
        x = x[:, :cut_off]
        return x


class LevenshteinDecoder(Decoder):
    def __init__(self, layer, n, output_embed_dim, tgt_vocab):
        super(LevenshteinDecoder, self).__init__(layer, n)

        # embeds the number of tokens to be inserted, max 256
        self.embed_mask_ins = Embedding(256, output_embed_dim * 2, None)

        # embeds the number of tokens to be inserted, max 256
        self.out_layer = Generator(output_embed_dim, tgt_vocab)

        # embeds either 0 or 1
        self.embed_word_del = Embedding(2, output_embed_dim, None)

    def extract_features(self, x, encoder_out,  x_mask):
        return self.forward(x, encoder_out, x_mask)

    def forward_mask_ins(self, encoder_out: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out,  x_mask)
        # creates pairs of consecutive words, so if the words are marked by their indices (0, 1, ... n):
        # [
        #   [0, 1],
        #   [1, 2],
        #   ...
        #   [n-1, n],
        # ]

        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return F.linear(features_cat, self.embed_mask_ins.weight)

    def forward_word_ins(self, encoder_out: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out,  x_mask)
        return self.out_layer(features)

    def forward_word_del(self, encoder_out: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out, x_mask)
        return F.linear(features, self.embed_word_del.weight)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

# class LevenshteinEncodeDecoder(EncoderDecoder):
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad, bos, eos, unk, criterion):
#         super(LevenshteinEncodeDecoder, self).__init__(encoder, decoder, src_embed, tgt_embed, generator)
#         self.pad = pad
#         self.eos = eos
#         self.bos = bos
#         self.unk = unk
#         self.criterion = criterion

#         # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)




#     # 全局视角 初始阶段的forward过程：先insert，然后delete
#     def forward(self, src: Tensor, tgt: Tensor):

#         # TODO: CHECK OUTPUTS OF ALL TGT-PRED PAIRS!!!
#         '''
#         从一个较高的层面来看，模型首次被调用时，通常会从一个非常粗略或空白的状态开始，
#         首先进行插入操作（生成占位符或初步的词汇），然后根据需要进行删除操作。
#         这个过程是指模型整体上如何开始处理一个全新的生成任务，或者在每次迭代之初的大致策略
#         '''


#         assert tgt is not None, "Forward function only supports training."

#         # encoding
#         encoder_out = self.encode(src)

#         #这里需要reorder，根据output
#         x = src
#         x_mask = BatchWithNoise.make_std_mask(x, pad=self.pad)



#         # generate training labels for insertion
#         # word_pred_tgt, word_pred_tgt_masks, ins_targets
#         word_pred_input, word_pred_tgt_masks, ins_targets = _get_ins_targets(x, tgt, self.pad, self.unk)
#         word_pred_tgt_subsequent_masks = BatchWithNoise.make_std_mask(word_pred_input, pad=self.pad)

#         ins_targets = ins_targets.clamp(min=0, max=255)  # for safe prediction
#         ins_masks = x[:, 1:].ne(self.pad)
#         ins_out = self.decoder.forward_mask_ins(encoder_out, self.tgt_embed(x), x_mask)

#         word_pred_out = self.decoder.forward_word_ins(encoder_out, self.tgt_embed(word_pred_input),
#                                                       word_pred_tgt_subsequent_masks)
#         # make online prediction
#         word_predictions = F.log_softmax(word_pred_out, dim=-1).max(2)[1]
#         word_predictions.masked_scatter_(~word_pred_tgt_masks, tgt[~word_pred_tgt_masks])
#         word_predictions_subsequent_mask = BatchWithNoise.make_std_mask(word_predictions, pad=self.pad)

#         # generate training labels for deletion
#         word_del_targets = _get_del_targets(word_predictions, tgt, self.pad)
#         word_del_out = self.decoder.forward_word_del(encoder_out, self.tgt_embed(word_predictions),
#                                                      word_predictions_subsequent_mask)
#         word_del_mask = word_predictions.ne(self.pad)

#         ins_loss = self.criterion(outputs=ins_out, targets=ins_targets, masks=ins_masks, label_smoothing=0.0)
#         word_pred_loss = self.criterion(outputs=word_pred_out, targets=tgt, masks=word_pred_tgt_masks,
#                                         label_smoothing=0.1)
#         word_del_loss = self.criterion(outputs=word_del_out, targets=word_del_targets,
#                                        masks=word_del_mask, label_smoothing=0.01)

#         # 用word_pred_out和tgt， 可以作一个KD 知识蒸馏

#         return {
#             "ins_loss": ins_loss,
#             "word_pred_loss": word_pred_loss,
#             "word_del_loss": word_del_loss,
#             "loss": ins_loss + word_pred_loss + word_del_loss
#         }
#     def encode(self, src: Tensor) -> Tensor:
#         return self.encoder(self.src_embed(src))

#     def decode(self, encoder_out: Tensor, x: Tensor, encoder_padding_mask, eos_penalty=0.0, max_ins_ratio=None, max_out_ratio=None) -> Tensor:

#         if max_ins_ratio is None:
#             max_ins_lens = x.new().fill_(255)
#             max_out_lens = x.new().fill_(255)
#         else:
#             src_lens = encoder_padding_mask.squeeze(1).sum(1)
#             max_ins_lens = (src_lens * max_ins_ratio).clamp(min=5).long()
#             max_out_lens = (src_lens * max_ins_ratio).clamp(min=10).long()

#         # delete words
#         # do not delete tokens if it is <s> </s>
#         can_del_word = x.ne(self.pad).sum(1) > 2
#         if can_del_word.sum() != 0:  # we cannot delete, skip
#             # x_mask = BatchWithNoise.make_std_mask(x, self.pad)
#             word_del_out = self.decoder.forward_word_del(
#                 _skip(encoder_out, can_del_word),
#                 self.tgt_embed(_skip(x, can_del_word)))

#             word_del_score = F.log_softmax(word_del_out, 2)
#             word_del_pred = word_del_score.max(-1)[1].bool()

#             _tokens = _apply_del_words(
#                 x[can_del_word],
#                 word_del_pred,
#                 self.pad,
#                 self.bos,
#                 self.eos,
#             )

#             x = _fill(x, can_del_word, _tokens, self.pad)

#         # insert placeholders
#         can_ins_mask = x.ne(self.pad).sum(1) < max_out_lens
#         if can_ins_mask.sum() != 0:
#             # x_mask = BatchWithNoise.make_std_mask(x, self.pad)
#             mask_ins_out = self.decoder.forward_mask_ins(
#                 _skip(encoder_out, can_ins_mask),
#                 self.tgt_embed(_skip(x, can_ins_mask)))

#             mask_ins_score = F.log_softmax(mask_ins_out, 2)
#             if eos_penalty > 0.0:
#                 mask_ins_score[:, :, 0] -= eos_penalty
#             mask_ins_pred = mask_ins_score.max(-1)[1]
#             limit_ins_pred = max_ins_lens[can_ins_mask, None].expand_as(mask_ins_pred)
#             mask_ins_pred = torch.min(mask_ins_pred, limit_ins_pred)

#             _tokens = _apply_ins_masks(
#                 x[can_ins_mask],
#                 mask_ins_pred,
#                 self.pad,
#                 self.unk,
#                 self.eos,
#             )
#             x = _fill(x, can_ins_mask, _tokens, self.pad)

#         # insert words
#         can_ins_word = x.eq(self.unk).sum(1) > 0
#         if can_ins_word.sum() != 0:
#             # x_mask = BatchWithNoise.make_std_mask(x, self.pad)
#             word_ins_out = self.decoder.forward_word_ins(
#                 _skip(encoder_out, can_ins_word),
#                 self.tgt_embed(_skip(x, can_ins_word)))

#             word_ins_score = F.log_softmax(word_ins_out, 2)
#             word_ins_pred = word_ins_score.max(-1)[1]

#             _tokens = _apply_ins_words(
#                 x[can_ins_word],
#                 word_ins_pred,
#                 self.unk,
#             )

#             x = _fill(x, can_ins_word, _tokens, self.pad)

#         # delete some unnecessary paddings
#         cut_off = x.ne(self.pad).sum(1).max()
#         x = x[:, :cut_off]
#         return x


# class LevenshteinDecoder(Decoder):
#     def __init__(self, layer, n, output_embed_dim, tgt_vocab):
#         super(LevenshteinDecoder, self).__init__(layer, n)

#         # embeds the number of tokens to be inserted, max 256
#         self.embed_mask_ins = Embedding(256, output_embed_dim * 2, None)

#         # embeds the number of tokens to be inserted, max 256
#         self.out_layer = Generator(output_embed_dim, tgt_vocab)

#         # embeds either 0 or 1
#         self.embed_word_del = Embedding(2, output_embed_dim, None)

#     def extract_features(self, x, encoder_out):

#         return self.forward(x, encoder_out)

#     def forward_mask_ins(self, encoder_out: Tensor, x: Tensor):
#         features = self.extract_features(x, encoder_out)
#         # creates pairs of consecutive words, so if the words are marked by their indices (0, 1, ... n):
#         # [
#         #   [0, 1],
#         #   [1, 2],
#         #   ...
#         #   [n-1, n],
#         # ]

#         features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
#         return F.linear(features_cat, self.embed_mask_ins.weight)

#     def forward_word_ins(self, encoder_out: Tensor, x: Tensor):
#         features = self.extract_features(x, encoder_out)
#         return self.out_layer(features)

#     def forward_word_del(self, encoder_out: Tensor, x: Tensor):
#         features = self.extract_features(x, encoder_out)
#         return F.linear(features, self.embed_word_del.weight)


# def Embedding(num_embeddings, embedding_dim, padding_idx):
#     m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
#     nn.init.constant_(m.weight[padding_idx], 0)
#     return m

"""### pointer network + LevT model

现在来看，pointer network只能改变encoder outputs的order；
此外levenshtein transformer的decoder input为noised target

用sinkhorn layer来对pointer中的index去重
"""

def LevenshteinTransformerModel(src_vocab, tgt_vocab, PAD, BOS, EOS, UNK, criterion, d_model=512, n=6, h=8, d_ff=2048,
                                dropout=0.0, input_dropout=0.1):
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, input_dropout)
    model = LevenshteinEncodeDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), n),
        LevenshteinDecoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout),
                           n=n, output_embed_dim=d_model, tgt_vocab=tgt_vocab),
        nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab),
        pad=PAD, bos=BOS, eos=EOS, unk=UNK,
        criterion=criterion
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

"""# Model Instance"""

class LabelSmoothingLoss(nn.Module):
    def __init__(self, factor=1.0, batch_multiplier=1.):
        super(LabelSmoothingLoss, self).__init__()
        self.factor = factor
        self.batch_multiplier = batch_multiplier

    def mean_ds(self, x: torch.Tensor, dim=None) -> torch.Tensor:
        return x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None, label_smoothing=0.0):
        if masks is not None:
            outputs = outputs[masks]
            targets = targets[masks]

        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets, reduction="none")

        else:  # soft-labels
            losses = F.kl_div(logits, targets, reduction="none")
            losses = losses.float().sum(-1).type_as(losses)

        nll_loss = self.mean_ds(losses)
        if label_smoothing > 0:
            loss = nll_loss * (1 - label_smoothing) - self.mean_ds(logits) * label_smoothing
        else:
            loss = nll_loss

        return loss * self.factor / self.batch_multiplier

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK = '<unk>'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# building shared vocabulary


pad_idx = tokenizer.convert_tokens_to_ids('<pad>')
bos_idx = tokenizer.convert_tokens_to_ids('<s>')
eos_idx = tokenizer.convert_tokens_to_ids('</s>')
unk_idx = tokenizer.convert_tokens_to_ids('<unk>')
print(f'Indexes -- PAD: {pad_idx}, EOS: {eos_idx}, BOS: {bos_idx}, UNK: {unk_idx}')



criterion = LabelSmoothingLoss()
criterion.to(device)


levt_model = LevenshteinTransformerModel(len(tokenizer.vocab), len(tokenizer.vocab),
                                    n=config['num_layers'],
                                    h=config['attn_heads'],
                                    d_model=config['model_dim'],
                                    dropout=config['dropout'],
                                    input_dropout=config['input_dropout'],
                                    d_ff=config['ff_dim'],
                                    criterion=criterion,
                                    PAD=pad_idx, BOS=bos_idx, EOS=eos_idx, UNK=unk_idx)



# weight tying  输入输出嵌入层权重共享
# pointer_model.src_embed[0].lookup_table.weight = pointer_model.tgt_embed[0].lookup_table.weight = levt_model.tgt_embed[0].lookup_table.weight

# levt_model.decoder.out_layer.lookup_table.weight = levt_model.tgt_embed[0].lookup_table.weight

# levt_model.src_embed[0].lookup_table.weight = levt_model.tgt_embed[0].lookup_table.weight
# levt_model.generator.lookup_table.weight = levt_model.tgt_embed[0].lookup_table.weight
# levt_model.decoder.out_layer.lookup_table.weight = levt_model.tgt_embed[0].lookup_table.weight



levt_model.to(device)


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



model_opt = NoamOpt(warmup_init_lr=config['warmup_init_lr'], warmup_end_lr=config['warmup_end_lr'],
                    warmup_updates=config['warmup'],
                    min_lr=config['min_lr'],
                    optimizer=torch.optim.Adam(levt_model.parameters(),
                                                lr=0,
                                                weight_decay=config['weight_decay'],
                                                betas=(config['beta_1'], config['beta_2']),
                                                eps=config['epsilon']))



# print(levt_model)

"""# Train"""

def save_checkpoint(model, optimizer,epoch, best_val_loss,filename="/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/PointerLevT_2.pth.tar"):
    """仅当模型更优时保存检查点"""

    checkpoint = {
        'state_dict1': model.state_dict(),  # 仅保存模型权重
        'optimizer': optimizer.optimizer.state_dict(),  # 保存优化器状态
        'epoch': epoch + 1,  # 保存当前训练轮数
        'best_val_loss': best_val_loss  # 保存最佳验证损失
    }
    torch.save(checkpoint, filename)


from tqdm import tqdm

def run_epoch(num_epochs, train_data, val_data, levt_model, opt):
    """
    Standard Training and Logging Function
    """
    best_val_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levt_model.to(device)
    criterion_pointer = nn.CrossEntropyLoss().to(device)


    for epoch in range(num_epochs):
        total_loss = 0
        levt_model.train()
        train_loader = tqdm(train_data, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)  # [batch_size, seq_length]
            attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_length]
            labels_tgt1 = batch['labels_tgt1'].to(device)
            labels_tgt2 = batch['labels_tgt2'].to(device)


            # seq_len = attention_scores.size(1)

            # pointer_loss = criterion(attention_scores.view(-1, seq_len), labels_tgt1.view(-1))

            out = levt_model(input_ids, labels_tgt1, labels_tgt2, 512)


            ins_loss = out['ins_loss'].item()
            word_pred_loss = out['word_pred_loss'].item()
            word_del_loss = out['word_del_loss'].item()
            loss_pointer = out['pointer_loss'].item()
            # print(loss_pointer)


            loss = 0.7*(out['ins_loss'] + out['word_pred_loss'] + out['word_del_loss']) + 0.3*(out['pointer_loss'])

            loss.backward()

            opt.step()
            opt.optimizer.zero_grad()
            current_lr = opt._rate
            # print(current_lr)

            total_loss += loss.item()
            train_loader.set_postfix(loss=total_loss / (i + 1), ins_loss=ins_loss, word_pred_loss=word_pred_loss, word_del_loss=word_del_loss, pointer_loss = loss_pointer)

        avg_loss = total_loss / len(train_data)
        print(f"Epoch: {epoch + 1} | Loss: {avg_loss} | loss_pointer:{loss_pointer} | ins_loss: {ins_loss} | word_pred_loss: {word_pred_loss} | word_del_loss: {word_del_loss}")



        levt_model.eval()
        total_val_loss = 0.0
        val_loader = tqdm(val_data, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch")

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)  # [batch_size, seq_length]
                attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_length]
                labels_tgt1 = batch['labels_tgt1'].to(device)
                labels_tgt2 = batch['labels_tgt2'].to(device)



                out = levt_model(input_ids , labels_tgt1, labels_tgt2, 512)

                ins_loss = out['ins_loss'].item()
                word_pred_loss = out['word_pred_loss'].item()
                word_del_loss = out['word_del_loss'].item()
                loss_pointer = out['pointer_loss'].item()

                loss = ins_loss + word_pred_loss + word_del_loss

                total_val_loss += loss


        avg_val_loss = total_val_loss / len(val_data)
        print(f"Epoch: {epoch + 1} | Val Loss: {avg_val_loss} | {loss_pointer}:loss_pointer | ins_loss: {ins_loss} | word_pred_loss: {word_pred_loss} | word_del_loss: {word_del_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(levt_model, opt, epoch + 1, best_val_loss)





run_epoch(50,train_loader,val_loader, levt_model, model_opt)

"""## 第二次训练"""

def load_checkpoint(model, optimizer):
    checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/PointerLevT_1.pth.tar')
    model.load_state_dict(checkpoint['state_dict1'])
    optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch} with best validation loss {best_val_loss}")
    return epoch, best_val_loss, optimizer, model

resume_epoch, best_val_loss, optimizer, model = load_checkpoint(levt_model, model_opt)

run_epoch(50,train_loader,val_loader, model, optimizer)

save_checkpoint(levt_model, model_opt, 1, 9.707)

best_val_loss = 9.707
levt_model.eval()
total_val_loss = 0.0
val_loader = tqdm(val_loader, desc=f"Validation Epoch {1}/{50}", unit="batch")

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)  # [batch_size, seq_length]
        attention_mask = batch['attention_mask'].to(device)  # [batch_size, seq_length]
        labels_tgt1 = batch['labels_tgt1'].to(device)
        labels_tgt2 = batch['labels_tgt2'].to(device)



        out = levt_model(input_ids , labels_tgt1, labels_tgt2, 512)

        ins_loss = out['ins_loss'].item()
        word_pred_loss = out['word_pred_loss'].item()
        word_del_loss = out['word_del_loss'].item()
        loss_pointer = out['pointer_loss'].item()

        loss = ins_loss + word_pred_loss + word_del_loss

        total_val_loss += loss


avg_val_loss = total_val_loss / len(val_loader)
print(f"Epoch: {1} | Val Loss: {avg_val_loss} | {loss_pointer}:loss_pointer | ins_loss: {ins_loss} | word_pred_loss: {word_pred_loss} | word_del_loss: {word_del_loss}")

if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    save_checkpoint(levt_model, optimizer, 1, best_val_loss)

def save_checkpoint(model, optimizer,epoch, best_val_loss,filename="/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/PointerLevT_1.pth.tar"):
    """仅当模型更优时保存检查点"""

    checkpoint = {
        'state_dict1': model.state_dict(),  # 仅保存模型权重
        'optimizer': optimizer.optimizer.state_dict(),  # 保存优化器状态
        'epoch': epoch + 1,  # 保存当前训练轮数
        'best_val_loss': best_val_loss  # 保存最佳验证损失
    }
    torch.save(checkpoint, filename)
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    save_checkpoint(levt_model, optimizer, 1, best_val_loss)

"""### inference/decoding"""

def load_checkpoint(model, optimizer):
    checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/PointerLevT_1.pth.tar')
    model.load_state_dict(checkpoint['state_dict1'])
    optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch} with best validation loss {best_val_loss}")
    return epoch, best_val_loss, optimizer, model

resume_epoch, best_val_loss, optimizer, model = load_checkpoint(levt_model, model_opt)

import time
def validate(model, test_loader, eos, bos, pad, max_decode_iter=10):
    total_time = 0
    with torch.no_grad():
        model.eval()


        hypotheses_tokenized = []
        references_tokenized = []

        hypotheses = []
        references = []



        for i, batch in enumerate(test_loader):

            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels_tgt2'].to(device)  # tgt
            refs = batch['target_text']
            tgt1 = batch['labels_tgt1'].to(device)


            decode_iter = 0

            encoder_out = model.encode(inputs)
            _, reordered_src, reordered_encoder_out = pointer_network(encoder_out,inputs, tgt1,d_model=512)
            prev_out = reordered_src
            while decode_iter < max_decode_iter:
            # start_time = time.time()

                out = model.decode(reordered_encoder_out, prev_out,attention_mask,
                                    max_ins_ratio=config['decoder_length_ratio'] if decode_iter == 0
                                    else config['decoder_insertion_ratio'],
                                    max_out_ratio=config['decoder_length_ratio'])
            # end_time = time.time()
            # total_time += end_time - start_time

        # print(f"Average time per batch: {total_time / len(test_loader)} seconds")
        # print(f"Total time: {total_time} seconds")
                # print(out)

                decode_iter += 1
                prev_out = out
            print("Reached maximum iterations.")
            # print(out)
            decoded_output = tokenizer.batch_decode(out, skip_special_tokens=True)
            # print('ref',refs)
            # print('decoded_output',decoded_output)
            for ref, pred in zip(refs, decoded_output):

                with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/pointer_real_ref_1.txt', 'a+') as f:
                    f.write(ref+'\n')
                with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/3_seq2edit/pointer_real_pred_1.txt', 'a+') as q:
                    q.write(pred+'\n')




validate(model, test_loader_real_tgt1, eos_idx, bos_idx, pad_idx, max_decode_iter=10)
