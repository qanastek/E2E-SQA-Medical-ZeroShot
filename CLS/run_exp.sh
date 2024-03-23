#!/bin/bash

source activate /workspace/spoken_medical_qa

torchrun --nproc_per_node=2 train_ssl.py hparams/whisper_ssl.yaml --experiment_name=whisper-medium --whisper_hub=openai/whisper-medium --encoder_dim=1024 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 
torchrun --nproc_per_node=2 train_ssl.py hparams/whisper_ssl.yaml --experiment_name=whisper-large-v2 --whisper_hub=openai/whisper-large-v2 --encoder_dim=1280 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 
torchrun --nproc_per_node=2 train_ssl.py hparams/ssl.yaml --experiment_name=wav2vec2-large --ssl_hub=facebook/wav2vec2-large --encoder_dim=1024 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 
torchrun --nproc_per_node=2 train_ssl.py hparams/ssl.yaml --experiment_name=data2vec-audio-large --ssl_hub=facebook/data2vec-audio-large --encoder_dim=1024 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 
torchrun --nproc_per_node=2 train_ssl.py hparams/ssl.yaml --experiment_name=hubert-large-ll60k --ssl_hub=facebook/hubert-large-ll60k --encoder_dim=1024 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 
torchrun --nproc_per_node=2 train_ssl.py hparams/whisper_ssl.yaml --experiment_name=whisper-small --whisper_hub=openai/whisper-small --encoder_dim=768 --number_of_epochs=5 --precision=fp16 --find_unused_parameters 