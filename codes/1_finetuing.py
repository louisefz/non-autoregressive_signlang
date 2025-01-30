# -*- coding: utf-8 -*-
# !pip install transformers
!pip install sacrebleu

import transformers
import torch
from transformers import AutoTokenizer, MBartForConditionalGeneration, AdamW, MT5ForConditionalGeneration




import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from sacrebleu.metrics import BLEU, CHRF, TER

import os

"""mask filling and translation through finetuning"""

from google.colab import drive
drive.mount('/content/gdrive')

class dataprocess4signgloss:
    def __init__(self, data_file):
        self.data_file = data_file


    def df(self):


        dfs_Sign_gloss = []
        dfs_Standard_text = []
        is_real = []



        with open(self.data_file,encoding='utf-8') as file_gloss:
            lines = file_gloss.readlines()



        for i in range(0, len(lines), 3):
            gloss = lines[i].strip()
            text = lines[i+1].strip()
            label = lines[i+2].strip()




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

gloss_file = "/content/gdrive/MyDrive/Colab Notebooks/sign_language/0_data_augmentation/monolingual_dataset/all_english_sentences.txt"

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

from torch.utils.data import Dataset

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

        # 目标文本用于计算损失，不需要attention_mask
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),  # 正确设置labels为目标文本的input_ids
            'source_text': source_text,
            'target_text': target_text
        }

import requests
requests.get('https://www.huggingface.co')

# tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25', src_lang="en_XX", tgt_lang="en_XX")   # de_De
tokenizer = AutoTokenizer.from_pretrained('google/mt5-large', src_lang="en_XX", tgt_lang="en_XX")
# model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-large')

# tweet_len=[]
# for text in data_df.Standard_text:
#     tokens=tokenizer.encode(text,max_length=50)
#     tweet_len.append(len(tokens))
# print(max(tweet_len))

# 准备数据集
train_dataset = TranslationDataset(X_train, y_train, tokenizer)
val_dataset = TranslationDataset(X_val, y_val, tokenizer)
test_dataset_all = TranslationDataset(X_test_all, y_test, tokenizer)
test_dataset_real = TranslationDataset(X_test_real, y_test_real, tokenizer)
test_dataset_fake = TranslationDataset(X_test_fake, y_test_fake, tokenizer)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader_all = DataLoader(test_dataset_all, batch_size=16, shuffle=False)
test_loader_real = DataLoader(test_dataset_real, batch_size=16, shuffle=False)
test_loader_fake = DataLoader(test_dataset_fake, batch_size=16, shuffle=False)

def train(model,num_epochs, train_loader, val_loader, device, learning_rate):
    bleu = BLEU()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        total_loss = 0.0
        all_predictions = []
        all_references = []

        model.train()

        train_loader = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            refs = batch['target_text']
            inputs_str = batch['source_text']



            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(loss)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()



            # 解码预测结果
            # predictions = model.generate(input_ids,max_length=50, min_length=20,length_penalty=1.0,num_beams=5,no_repeat_ngram_size=2)
            # decoded_preds = [' '.join(tokenizer.convert_ids_to_tokens(id,skip_special_tokens=True)) for id in predictions][0]
            # print(refs)
            # decoded_refs = [' '.join(tokenizer.tokenize(ref)) for ref in refs]
            # print(f'Input = {inputs_str}, Ref = {decoded_refs}, Prediction = {decoded_preds}')

            # 计算BLEU分数
            # all_predictions.extend(decoded_preds)
            # all_references.extend([[ref] for ref in decoded_refs])

        average_train_loss = total_loss / len(train_loader)
        # bleu_score = bleu.corpus_score(all_predictions, all_references)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss}")
        # print(f"Epoch {epoch+1}/{num_epochs}: BLEU score = {bleu.score}")





        model.eval()
        total_val_loss = 0.0

        all_predictions_val = []
        all_references_val = []

        with torch.no_grad():
            val_loader = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            for batch in val_loader:
                input_ids_val = batch['input_ids'].to(device)
                attention_mask_val = batch['attention_mask'].to(device)
                labels_val = batch['labels'].to(device)
                refs_val = batch['target_text']

                outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val, labels=labels_val)
                loss_val = outputs_val.loss
                total_val_loss += loss_val.item()

                predictions_val = model.generate(input_ids_val)
                decoded_preds_val = tokenizer.batch_decode(predictions_val, skip_special_tokens=True)
                decoded_refs_val = refs_val

                all_predictions.extend(decoded_preds_val)
                all_references.extend([[ref] for ref in decoded_refs_val])


                # 在这里根据需要执行其他验证操作
        average_val_loss = total_val_loss / len(val_loader)
        # bleu_score_val = bleu.corpus_score(all_predictions_val, all_references_val)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_val_loss}")
        # print(f"Epoch {epoch+1}/{num_epochs}: BLEU score_Val = {bleu.score}")

def save_checkpoint(model, optimizer,epoch, best_val_loss,filename="/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/model_checkpoint.pth.tar"):
    """仅当模型更优时保存检查点"""

    checkpoint = {
        'state_dict': model.state_dict(),  # 仅保存模型权重
        'optimizer': optimizer.state_dict(),  # 保存优化器状态
        'epoch:': epoch + 1,  # 保存当前训练轮数
        'best_val_loss': best_val_loss  # 保存最佳验证损失
    }
    torch.save(checkpoint, filename)

def train(model, num_epochs, train_loader, val_loader, device, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # 每个epoch后降低学习率
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in train_loader_tqdm:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_postfix({'loss': loss.item()})

        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr}")
        average_train_loss = total_train_loss / len(train_loader)
        train_loader_tqdm.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)

        # 更新最佳验证损失并保存检查点
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_checkpoint(model, optimizer,epoch, best_val_loss)

        print(f"Validation Loss: {average_val_loss:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= model.to(device)
train_loader = train_loader
val_loader = val_loader
learning_rate = 1e-5
num_epochs = 20

train(model,num_epochs, train_loader, val_loader, device, learning_rate)

# 在测试集上评估模型
import time

checkpoint = torch.load('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/model_checkpoint_MT5.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda")
model.eval()
model.to(device)
total_inference_time = 0
with torch.no_grad():
    for i,batch in enumerate(test_loader_real):
        print(i)
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        refs = batch['target_text']
        # outputs = model(inputs, labels=labels)

        # 解码预测结果
        # predictions = model.generate(inputs)
        # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # print("refs:", refs)
        # print("decoded_preds:", decoded_preds)

        # for ref, pred in zip(refs, decoded_preds):
        # for ref in refs:
        #     with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_ref_real.txt', 'a+') as f:
        #         f.write(ref+'\n')
        #     # with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_pred_real.txt', 'a+') as f:
        #     #     f.write(pred+'\n')
        #     # print(f'Ref = {ref}, Prediction = {pred}')
        #     print(f'Ref = {ref}')
        start_time = time.time()

        predictions = model.generate(inputs)
        end_time = time.time()

        inference_time = end_time - start_time
        total_inference_time += inference_time
    print(total_inference_time)
    print(f"Average inference time: {total_inference_time / len(test_loader_real)} seconds")
        # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # decoded_preds = [' '.join(tokenizer.convert_ids_to_tokens(id,skip_special_tokens=True)) for id in predictions][0]
        # print(refs)
        # decoded_refs = [' '.join(tokenizer.tokenize(ref)) for ref in refs]
        # print(f'Input = {inputs_str}, Ref = {decoded_refs}, Prediction = {decoded_preds}')

    # for batch in test_loader_real:
    #     inputs = batch['input_ids']
    #     labels = batch['labels']

    #     outputs = model(inputs, labels=labels)


    # for batch in test_loader_fake:
    #     inputs = batch['input_ids']
    #     labels = batch['labels']

    #     outputs = model(inputs, labels=labels)

"""##找real_prediction的文本"""

dir_all_ref = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_ref_all.txt'
dir_all_pred = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_pred_all.txt'
dir_ref_real = '/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_ref_real.txt'

with open(dir_all_ref, 'r') as f:
    all_ref = f.readlines()

with open(dir_all_pred, 'r') as q:
    all_pred = q.readlines()

with open(dir_ref_real, 'r') as w:
    ref_real = w.readlines()

# i = 0
# for ref, pred in zip(all_ref, all_pred):
#     for realref in ref_real:
#         # i += 1
#         # print(i)
#         if realref == ref:
#             # print(realref)
#             # print(ref)
#             with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/mbart_real_real_pred.txt', 'a+') as t:
#                 t.write(realref + pred)

"""### target data的特性

1. source and target length: ref和pred(real and all)
2. accuracy: 多少个单词一样，tokenized accuracy
3. order distance
4. BLEU：n-gram:1, 2, 3, 4

"""

from transformers import AutoTokenizer
from sacrebleu import corpus_bleu
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

def nltk_bleu(all_text1, all_text2):
    sent1_list_ref = [[sent.strip().split()] for sent in all_text1]
    sent2_list_pred = [sent.strip().split() for sent in all_text2]

    bleu_score = nltk_corpus_bleu(sent1_list_ref, sent2_list_pred)
    return bleu_score

"""### length"""

print(length(all_ref))
print(length(all_pred))

with open('/content/gdrive/MyDrive/Colab Notebooks/sign_language/3_model_save/1_seq2seq_finetune/MT5_real_real_pred.txt') as t:
    german_sentences = t.readlines()


no_duplicate_ref = []
no_duplicate_pred = []
for i in range(0, len(german_sentences), 2):

    real_gloss = german_sentences[i].strip()
    tgt1 = german_sentences[i+1].strip()

    if real_gloss not in no_duplicate_ref: # and real_gloss != '' and tgt1 != '':
        no_duplicate_ref.append(real_gloss)
        no_duplicate_pred.append(tgt1)

print(length(no_duplicate_ref))
print(length(no_duplicate_pred))


acc_all = 0
acc_token_all = 0
lev_distance_all = 0
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

print(len(no_duplicate_ref))
print(len(no_duplicate_pred))

print(nltk_bleu(no_duplicate_ref, no_duplicate_pred))
# print(sacre_bleu(no_duplicate_ref, no_duplicate_pred))

acc_all = 0
acc_token_all = 0
lev_distance_all = 0


for sent1, sent2 in zip(all_ref, all_pred):
    # print(sent1)
    # print(sent2)
    acc = accuracy(sent1, sent2)
    acc_token = accurcy_token(sent1, sent2)
    lev_distance = levenshtein_two_matrix_rows(sent1, sent2)
    acc_all += acc
    acc_token_all += acc_token
    lev_distance_all += lev_distance

acc_avg = acc_all/len(all_ref)
acc_token_avg = acc_token_all/len(all_ref)
lev_distance_avg = lev_distance_all/len(all_ref)


print(acc_avg, acc_token_avg, lev_distance_avg)

print(len(all_ref))
print(len(all_pred))

print(nltk_bleu(all_ref, all_pred))
