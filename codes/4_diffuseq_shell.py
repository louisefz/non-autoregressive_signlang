# -*- coding: utf-8 -*-


# ! pip install -r requirements.txt
!pip install blobfile
! pip install wandb
!pip install datasets
# ! pip install torchmetrics
# !pip install bert_score
# import shutil
from google.colab import drive
import shutil

!git clone https://github.com/louisefz/diffuseq_sign.git

drive.mount('/content/drive')

src = '/content/drive/MyDrive/Colab Notebooks/sign_language/4_optimization/diffuseq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20240722-14:46:50'
dst = '/content/diffuseq_sign/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20240722-14:46:50'

# Use shutil.copytree to copy a directory
shutil.copytree(src, dst, dirs_exist_ok=True)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/diffuseq_sign/scripts
! bash /content/diffuseq_sign/scripts/train.sh

# 假设文件example.txt在当前工作目录中
source_path = '/content/diffuseq_sign/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20240722-14:46:50'

# 目标路径：将文件保存到Google Drive中的my_folder文件夹
destination_path = '/content/drive/MyDrive/Colab Notebooks/sign_language/4_optimization/diffuseq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20240722-14:46:50'

# 将文件从当前目录复制到Google Drive
# shutil.copy(source_path, destination_path)
shutil.copytree(source_path, destination_path)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/diffuseq_sign/scripts
!bash /content/diffuseq_sign/scripts/run_decode.sh

! /content/DiffuSeq/scripts
!python /content/DiffuSeq/scripts/eval_seq2seq.py  --folder /content/DiffuSeq/generate_outputs --mbr

!ls /content/DiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test-qqp20240720-17:09:04

