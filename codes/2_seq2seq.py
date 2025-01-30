# -*- coding: utf-8 -*-

# !pip install transformers
# !pip install rouge-score
# !pip install sacrebleu
# !pip install "sacrebleu[ko]" # 支持韩语的tokenizer

from google.colab import drive
drive.mount('/content/gdrive')

import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartConfig,  AutoTokenizer

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA

from copy import deepcopy
from torch import Tensor
import math
# from rouge_score import rouge_scorer
# from sacrebleu.metrics import BLEU, CHRF, TER

"""# 2 Data Preprocessing"""

class dataprocess4signgloss:
    def __init__(self, data_file):
        self.data_file = data_file


    def df(self):


        dfs_Sign_gloss = []
        dfs_Standard_text = []
        is_real = []



        with open(self.data_file,encoding='utf-8') as file_gloss:
            lines = file_gloss.readlines()



        for i in range(0, len(lines), 4):
            gloss = lines[i].strip()
            text = lines[i+2].strip()
            label = lines[i+3].strip()




            dfs_Sign_gloss.append(gloss)
            dfs_Standard_text.append(text)
            is_real.append(label)
        data = {
            'Sign_gloss': dfs_Sign_gloss,
            'Standard_text':dfs_Standard_text,
            'is_real':is_real
        }

        df_data = pd.DataFrame(data)


        return df_data

gloss_file = "/content/gdrive/MyDrive/Colab Notebooks/sign_language/0_data_augmentation/monolingual_dataset/english_sentences_good_large_dataset.txt"

data = dataprocess4signgloss(gloss_file)

# print(data.df())

data_df = data.df()
print(data_df.head(5))

# X = data_df[['Sign_gloss']]
# y = data_df['Standard_text']

# X_train, X_test_val, y_train, y_test_val = train_test_split(X,y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

X = data_df[['Sign_gloss', 'is_real']]  # 包括 'is_real' 列以便于分割
y = data_df['Standard_text']
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=data_df['is_real'])
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=X_test_val['is_real'])




# print(y_train.to_string(index=False))
is_real_counts = X_train['is_real'].value_counts()
print(is_real_counts)
is_real_counts1 = X_val['is_real'].value_counts()
print(is_real_counts1)
is_real_counts2 = X_test['is_real'].value_counts()
print(is_real_counts2)

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

y_test_real = y_test.loc[X_test_real.index]
y_test_fake = y_test.loc[X_test_fake.index]

# print(X_test_real.head(5))
# print(y_test_real.head(5))
# print(X_test_fake.head(5))
# print(y_test_fake.head(5))
# print(X_test_fake.head(5))
# print(X_test_real_all.head(5))

class TranslationDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length=50):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # 从数据集中获取源文本和目标文本
        source_text = str(self.X.iloc[idx][0])#.split("    ")[1].split("\n")[0]
        target_text = str(self.y.iloc[idx])

        # 对源文本和目标文本进行Tokenization
        # 注意: 我们为源文本和目标文本使用不同的处理方法，尤其是当目标文本被用作 labels 时
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )


        # # 目标文本用于计算损失，不需要attention_mask
        # with self.tokenizer.as_target_tokenizer():
        #     target_encoding = self.tokenizer(
        #         target_text,
        #         max_length=self.max_length,
        #         padding='max_length',
        #         truncation=True,
        #         return_tensors="pt"
        #     )
        return {
        'src_input_ids': source_encoding['input_ids'].squeeze(0),
        'src_attention_mask': source_encoding['attention_mask'].squeeze(0),
        'tgt_input_ids': target_encoding['input_ids'].squeeze(0),
        'labels': target_encoding['input_ids'].squeeze(0),  # 正确设置labels为目标文本的input_ids
        'source_text': source_text,
        'target_text': target_text
        }

tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang="en_XX", tgt_lang="en_XX")   # de_De

# 准备数据集
train_dataset = TranslationDataset(X_train, y_train, tokenizer)
val_dataset = TranslationDataset(X_val, y_val, tokenizer)
test_dataset_all = TranslationDataset(X_test_all, y_test, tokenizer)
test_dataset_real = TranslationDataset(X_test_real, y_test_real, tokenizer)
test_dataset_fake = TranslationDataset(X_test_fake, y_test_fake, tokenizer)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader_all = DataLoader(test_dataset_all, batch_size=64, shuffle=False)
test_loader_real = DataLoader(test_dataset_real, batch_size=64, shuffle=False)
test_loader_fake = DataLoader(test_dataset_fake, batch_size=64, shuffle=False)

# for batch in train_loader:
#     print(batch['tgt_input_ids'].size(1))

"""# 3 Configure Teacher and Student
teacher model: BART, MBART, T5, mT5，用已经


student model: Transformer


1.   **hyperparamters define:** hidden layers, hidden states
2.   **loss define:** hard, soft, hidden states
3.   whether to add embedding transfer to accelerate the weight tuning




"""

class Teachermodel(nn.Module):
    def __init__(self, teacher_models_frame, teacher_models_pt,teacher_models_checkpoint):
        super(Teachermodel,self).__init__()

        self.input = input
        # self.teacher_tokenizer = tokenizer
        self.teacher_model = teacher_models_frame.from_pretrained(teacher_models_pt)
        checkpoint = torch.load(teacher_models_checkpoint)
        self.teacher_model.load_state_dict(checkpoint['state_dict'])



    def hyperparameter_check(self):
        config = self.teacher_model.config
        print("Embedding Size:", config.d_model)
        print("Vocabulary Size:", config.vocab_size)
        print("Number of Layers:", config.num_hidden_layers)
        print("Number of Hidden States per Layer:", config.hidden_size)
        return config

    def forward(self,input_ids):
        # tokenized_input = self.teacher_tokenizer(input, return_tensor="pt", max_length=256, truncation=True)
        outputs = self.teacher_model(input_ids)
        # decoder hidden state
        # last_decoder_hidden_state = outputs.last_hidden_state
        # logit
        logits = outputs.logits
        # print("Decoder Last Hidden State Shape:", last_decoder_hidden_state.shape)
        # print("Logits Shape:", logits.shape)

        # label
        # decoded_output = self.teacher_tokenizer.decode(logits.argmax(-1)[0], skip_special_tokens=True)
        # print("Decoded Output:", decoded_output)
        # print("mbart_logits:", logits)
        # print(logits.shape)
        return logits


class Studentmodel(nn.Module):
    """
    vanilla transformer is built here

    encoder: ** layers, ** hidden units    # use other baselines
    decoder: ** layers, ** hidden units    # use other baselines

    """
    def __init__(self, d_model = 512, vocab_size = 250027, n_layers = 6 , heads = 8, d_ff = 2048, dropout=0.0, loss_fn = nn.CrossEntropyLoss()):
        super(Studentmodel,self).__init__()



        self.vocab_size = vocab_size
        # self.loss_fn = loss_fn
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.model = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=d_ff, dropout=dropout,batch_first=True)
        # self.fc_out = nn.Linear(d_model, vocab_size)

        custom_config = MBartConfig(
            d_model=d_model,        # 较小的隐藏层维度
            vocab_size=vocab_size,   # 标准词汇表大小
            encoder_layers=n_layers,   # 减少的encoder层数
            decoder_layers=n_layers,   # 减少的decoder层数
            encoder_attention_heads=heads,   # attention heads的数量
            decoder_attention_heads=heads,
            encoder_ffn_dim=d_ff,   # feedforward层的维度
            decoder_ffn_dim=d_ff
        )
        student_model = MBartForConditionalGeneration(config=custom_config)
        self.student_model = student_model
        self.loss_fn = nn.CrossEntropyLoss()


    def hyperparameter_check(self):
        config = self.student_model.config
        print("Embedding Size:", config.d_model)
        print("Vocabulary Size:", config.vocab_size)
        print("Number of Layers:", config.num_hidden_layers)
        print("Number of Hidden States per Layer:", config.hidden_size)
        return config

    def forward(self,input_ids,attention_mask,labels):
        outputs = self.student_model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        # print(logits.shape)
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return logits,loss
    def decode(self,input_ids,attention_mask):
        outputs = self.student_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_token_ids = torch.argmax(logits, dim=-1)
        decoded_output = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        return decoded_output

student_model = Studentmodel()

student_model.hyperparameter_check()

teacher_models_frame = MBartForConditionalGeneration
teacher_models_pt = 'facebook/mbart-large-cc25'
teacher_models_checkpoint = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/model_checkpoint.pth.tar'


teacher_model = Teachermodel(teacher_models_frame, teacher_models_pt,teacher_models_checkpoint)

teacher_model.hyperparameter_check()

"""# 5 Pretraining Student with KD

在pretrain的过程中加上KD
"""

def save_checkpoint(model, optimizer,epoch, best_val_loss,filename="/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/KD_checkpoint_large_dataset1111.pth.tar"):
    """仅当模型更优时保存检查点"""

    checkpoint = {
        'state_dict': model.state_dict(),  # 仅保存模型权重
        'optimizer': optimizer.state_dict(),  # 保存优化器状态
        'epoch:': epoch + 1,  # 保存当前训练轮数
        'best_val_loss': best_val_loss  # 保存最佳验证损失
    }
    torch.save(checkpoint, filename)


def knowledge_distillation_train(teacher, student, train_loader, val_loader, best_val_loss = -math.inf, epochs = 50, learning_rate=1e-4, weight = 0.6, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)  # 每个epoch后降低学习率
    # fn_loss = torch.nn.CrossEntropyLoss().to(device)
    # kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(device)
    teacher.to(device)
    student.to(device)
    teacher.eval()  # Ensure the teacher is in eval mode for inference
    # epochs = int(epochs)

    best_val_loss = best_val_loss
    for epoch in range(epochs):
        student.train()  # Set student model to training mode
        total_loss = 0.0

        # Training loop with tqdm progress bar
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')
        for batch in train_bar:
            # print(batch)
            src_inputs,src_mask,labels = batch['src_input_ids'].to(device), batch['src_attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                # Ensure attention mask is used in model forward pass
                teacher_logits = teacher(src_inputs)

            student_logits,label_loss = student(src_inputs,src_mask,labels)

            kl_loss = torch.nn.functional.kl_div(
                input=torch.nn.functional.log_softmax(student_logits, dim=-1),
                target=torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction='batchmean')


            loss = weight * kl_loss + (1 - weight) * label_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix({'label_loss:':label_loss,'kl_loss':kl_loss,'loss': loss.item()})


        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr}")
        avg_loss = total_loss / len(train_loader)
        train_bar.set_description(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")




        # 验证阶段
        student.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                src_input_ids = batch['src_input_ids'].long().to(device)
                src_attention_mask = batch['src_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                student_logits,val_loss = student(src_input_ids,src_attention_mask,labels)

                with torch.no_grad():
                    # Ensure attention mask is used in model forward pass
                    teacher_logits = teacher(src_input_ids)

                kl_loss_val = torch.nn.functional.kl_div(
                    input=torch.nn.functional.log_softmax(student_logits, dim=-1),
                    target=torch.nn.functional.softmax(teacher_logits, dim=-1),
                    reduction='batchmean')

                loss = weight * kl_loss_val + (1 - weight) * val_loss

                total_val_loss += loss.item()


        average_val_loss = total_val_loss / len(val_loader)

        # 更新最佳验证损失并保存检查点
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_checkpoint(student, optimizer,epoch, best_val_loss)

        print(f"Validation Loss: {average_val_loss:.4f}")

"""用L4训练的，第14个epoch，最好的loss为5.3897，每个epoch是55分钟"""

knowledge_distillation_train(teacher_model, student_model, train_loader, val_loader)

"""## Further pretrain with larger dataset"""

checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/KD_checkpoint.pth.tar')
student_model.load_state_dict(checkpoint['state_dict'])
# start_epoch = checkpoint['epoch:']
# loss = checkpoint['best_val_loss']

# print("start_epoch:",start_epoch)
# print("loss:",loss)

"""### 这是第二次训练的结果"""

# knowledge_distillation_train(teacher_model, student_model, train_loader, val_loader)

"""### 这是第三次训练"""

def load_checkpoint(model):
    checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/KD_checkpoint_large_dataset.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch:']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded. Resuming from epoch {epoch} with best validation loss {best_val_loss}")
    return epoch, best_val_loss, model

resume_epoch, best_val_loss, student_model = load_checkpoint(student_model)

knowledge_distillation_train(teacher_model, student_model, train_loader, val_loader, best_val_loss)

"""## Inference on test set"""

checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/KD_checkpoint_large_dataset.pth.tar')
student_model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda")
student_model.to(device)
student_model.eval()

# model.eval()
# model.to(device)

import time

total_time = 0
device = torch.device("cuda")
student_model.eval()
student_model.to(device)
with torch.no_grad():
    for i,batch in enumerate(test_loader_real):
        print(i)
        inputs = batch['src_input_ids'].to(device)
        attention_mask = batch['src_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        refs = batch['target_text']
        # outputs = model(inputs, labels=labels)

        # 解码预测结果

        start_time = time.time()

        decoded_preds = student_model.decode(inputs, attention_mask)

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
    print(f"Average inference time: {total_time / len(test_loader_real):.4f} seconds")
    print(f"Total inference time: {total_time:.4f} seconds")

        # print(f"Inference time: {end_time - start_time:.4f} seconds")
        # print("refs:", refs)
        # print("decoded_preds:", decoded_preds)

        # for ref, pred in zip(refs, decoded_preds):

        #     with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_ref_all.txt', 'a+') as f:
        #         f.write(ref+'\n')
        #     with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_pred_all.txt', 'a+') as f:
                # f.write(pred+'\n')
            # print(f'Ref = {ref}, Prediction = {pred}')

"""##找real_prediction的文本"""

dir_all_ref = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_ref_all.txt'
dir_all_pred = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_pred_all.txt'
dir_ref_real = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_ref_real.txt'
dir_pred_real = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/2_seq2seq_KD/small_pred_real.txt'

with open(dir_all_ref, 'r') as f:
    all_ref = f.readlines()

with open(dir_all_pred, 'r') as q:
    all_pred = q.readlines()

with open(dir_ref_real, 'r') as w:
    ref_real = w.readlines()

with open(dir_pred_real, 'r') as e:
    pred_real = e.readlines()

"""### target data的特性

1. source and target length: ref和pred(real and all)
2. accuracy: 多少个单词一样，tokenized accuracy
3. order distance
4. BLEU：n-gram:1, 2, 3, 4

"""

from transformers import AutoTokenizer

from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang="en_XX", tgt_lang="en_XX")

def length(text):
    sentence_len_list = [len(sent.strip().split()) for sent in text]
    avg_sent_len = sum(sentence_len_list)/len(sentence_len_list)
    return avg_sent_len


def accuracy(sent1, sent2):
    sent1_set = set(sent1.strip().split())
    sent2_set = set(sent2.strip().split())
    common_words_len = len(sent1_set.intersection(sent2_set))
    return common_words_len/len(sent1.strip().split())

def accurcy_token(sent1, sent2):

    sent_tokens1 = tokenizer.tokenize(sent1)
    sent_tokens2 = tokenizer.tokenize(sent2)
    common_tokens_len = len(set(sent_tokens1).intersection(set(sent_tokens2)))
    return common_tokens_len/len(sent_tokens1)


def levenshtein_two_matrix_rows(sent1, sent2):
    # Get the lengths of the input strings
    # print(sent1)
    # print(sent2)
    sent1_list = sent1.strip().split()
    sent2_list = sent2.strip().split()
    m = len(sent1_list)
    n = len(sent2_list)

    # Initialize two rows for dynamic programming
    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = i

        for j in range(1, n + 1):
            if sent1_list[i - 1] == sent2_list[j - 1]:
                # Characters match, no operation needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Choose the minimum cost operation
                curr_row[j] = 1 + min(
                    curr_row[j - 1],  # Insert
                    prev_row[j],      # Remove
                    prev_row[j - 1]    # Replace
                )

        # Update the previous row with the current row
        prev_row = curr_row.copy()

    # The final element in the last row contains the Levenshtein distance
    return curr_row[n]

def levenshtein_token(sent1, sent2):
    sent_tokens1 = tokenizer.tokenize(sent1)
    sent_tokens2 = tokenizer.tokenize(sent2)
    return levenshtein_two_matrix_rows(sent_tokens1, sent_tokens2)

def nltk_bleu(all_text1, all_text2):
    sent1_list_ref = [[sent.strip().split()] for sent in all_text1]
    sent2_list_pred = [sent.strip().split() for sent in all_text2]

    bleu_score = nltk_corpus_bleu(sent1_list_ref, sent2_list_pred)
    return bleu_score

"""### length"""

print(length(all_ref))
print(length(all_pred))
print(length(ref_real))
print(length(pred_real))

no_duplicate_ref = []
no_duplicate_pred = []
for ref, pred in zip(ref_real,pred_real):

    real_gloss = ref.strip().lower()
    tgt1 = pred.strip().lower()

    if real_gloss not in no_duplicate_ref: # and real_gloss != '' and tgt1 != '':
        no_duplicate_ref.append(real_gloss)
        no_duplicate_pred.append(tgt1)

print(length(no_duplicate_ref))
print(length(no_duplicate_pred))


acc_all = 0
acc_token_all = 0
lev_distance_all = 0
lev_tok_all = 0
for sent1, sent2 in zip(no_duplicate_ref, no_duplicate_pred):
    # print(sent1)
    # print(sent2)
    acc = accuracy(sent1, sent2)
    acc_token = accurcy_token(sent1, sent2)
    lev_distance = levenshtein_two_matrix_rows(sent1, sent2)

    acc_all += acc
    acc_token_all += acc_token
    lev_distance_all += lev_distance


acc_avg = acc_all/len(no_duplicate_ref)
acc_token_avg = acc_token_all/len(no_duplicate_ref)
lev_distance_avg = lev_distance_all/len(no_duplicate_ref)



print(acc_avg, acc_token_avg, lev_distance_avg)
# print(acc_token_avg)

print(len(no_duplicate_ref))
print(len(no_duplicate_pred))

print(nltk_bleu(no_duplicate_ref, no_duplicate_pred))
# print(sacre_bleu(no_duplicate_ref, no_duplicate_pred))
