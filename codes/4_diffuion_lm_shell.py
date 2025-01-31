# -*- coding: utf-8 -*-
# ! pip install -r requirements.txt
!pip install blobfile
! pip install wandb
!pip install datasets
!pip install mpi4py

! git clone https://github.com/Diffusion-LM.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/Diffusion-LM/improved-diffusion
! mkdir /content/Diffusion-LM/diffusion_models

! python /content/Diffusion-LM/improved-diffusion/scripts/run_train.py \
 --diff_steps 2000 \
 --model_arch transformer \
 --lr 0.0001 \
 --lr_anneal_steps 200000  \
 --seed 102 \
 --noise_schedule sqrt \
 --in_channel 16 \
 --modality e2e-tgt \
 --submit no \
 --padding_mode block \
 --bsz 64 \
 --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " \
 --notes xstart_e2e

! mkdir /content/Diffusion-LM/generation_outputs

model_path = '/content/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e.sh'
! python scripts/batch_decode.py model_path -1.0 ema

! python scripts/text_sample.py --model_path /content/Diffusion-LM/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e/ema_0.9999_000000.pt --out_dir /content/Diffusion-LM/generation_outputs
