import os
import json

import whisper
from tqdm import tqdm

for whisper_version in ["large-v2","medium.en"]:

    print(whisper_version)

    model = whisper.load_model(whisper_version)

    bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

    for subset in ["test","train"]:

        print(subset)

        if subset == "test":
            dir_audios = "audios_wav_ffmpeg_concat"
        else:
            dir_audios = "audios_wav_ffmpeg_train_concat"
        
        data = {}
        
        for filepath in tqdm(os.listdir(dir_audios)):

            if subset not in filepath:
                continue

            new_file_name = filepath
            for mmlu_c in bad_corpus_names:
                new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
            
            corpus_name = new_file_name.split("_")[0].replace("-","_")

            transcription = model.transcribe(f"./{dir_audios}/{filepath}")["text"]

            if corpus_name not in data:
                data[corpus_name] = []
            
            data[corpus_name].append(transcription)
        
        for _corpus in data:

            with open(f"./transcription/{_corpus}_{subset}.json", 'w') as f:
                json.dump(data[_corpus], f, indent=4)
