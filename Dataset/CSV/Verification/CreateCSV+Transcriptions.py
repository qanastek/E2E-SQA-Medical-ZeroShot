import os
import json
import shutil

import pandas as pd
from tqdm import tqdm
import soundfile as sf

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

for CURRENT_SUBSET in ["train","test"]:

    if CURRENT_SUBSET == "test":
        json_path = "./json"
    else:
        json_path = "./json_train"

    for file_name in os.listdir(json_path):

        current_ok = 0

        if has_numbers(file_name):
            continue
        
        print(file_name)
        
        f_in = open(f"{json_path}/{file_name}","r")
        data = json.load(f_in)
        f_in.close()

        new_file_name = file_name
        for mmlu_c in bad_corpus_names:
            new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
        
        splitted = new_file_name.split("_")
        corpus_name = splitted[0].replace("-","_")
        subset = splitted[1].replace(".json","")

        if subset != CURRENT_SUBSET:
            continue

        all_transcription = {}

        if subset == "test":
            
            for size in ["small","medium","large-v2"]:
                
                transcription_path = f"./transcription/{corpus_name}_{subset}_{size}.json"

                f_data = open(transcription_path, "r")
                data_size = json.load(f_data)
                f_data.close()

                all_transcription[size] = data_size

        header = ["ID", "ID_original_dataset", "duration", "duration_no_answer", "wav", "wav_no_answer", "spk_id", "wrd", "small", "medium", "large-v2", "class"]
        df_data_classification = []

        for idx, d in enumerate(tqdm(data)):

            if subset == "test":
                dir_audio = "audios_wav_ffmpeg_concat"
                dir_audio_no_answer = "audios_wav_ffmpeg_concat"
                end_sentence = "The correct answer is option "
            else:
                dir_audio = "audios_wav_ffmpeg_train_concat"
                dir_audio_no_answer = "audios_wav_ffmpeg_train_concat_FT"
                end_sentence = f"The correct answer is option {d['answer'].upper()}"

            new_path_audio = d["audio_path"].replace("./audios/",f"./{dir_audio}/")
            
            if os.path.isfile(new_path_audio) == False:
                if CURRENT_SUBSET == "test":
                    print("Error: No file found! - ", d["identifier"])
                    print()
                print(new_path_audio)
                continue
            
            current_ok += 1

            audio_f = sf.SoundFile(new_path_audio)

            audio_path = new_path_audio.replace(f"./{dir_audio}/", f"/local_disk/ether/ylabrak/SpokenMedicalQA/{dir_audio}/")
            audio_path_no_answer = new_path_audio.replace(f"./{dir_audio}/", f"/local_disk/ether/ylabrak/SpokenMedicalQA/{dir_audio_no_answer}/")

            audio_f_no_answer = sf.SoundFile(new_path_audio.replace(f"./{dir_audio}/", f"./{dir_audio_no_answer}/"))

            if subset == "test":
                file_name_audio = audio_path.split("/")[-1]
                transcript_small = all_transcription["small"][file_name_audio]
                transcript_medium = all_transcription["medium"][file_name_audio]
                transcript_largev2 = all_transcription["large-v2"][file_name_audio]
            
            elif subset == "train":
                transcript_small = ""
                transcript_medium = ""
                transcript_largev2 = ""

            df_data_classification.append([
                f"{idx}",
                f"{d['identifier']}",
                audio_f.frames / audio_f.samplerate,
                audio_f_no_answer.frames / audio_f_no_answer.samplerate,
                audio_path,
                audio_path_no_answer,
                d["speaker"],

                d["prompt_no_answer"].replace("\n"," ").replace("\t"," ") + " " + end_sentence,
                transcript_small,
                transcript_medium,
                transcript_largev2,

                d["answer"]
            ])

        df_classification = pd.DataFrame(df_data_classification, columns=header)
        df_classification.to_csv(f"./csv+transcripts/{corpus_name}_{subset}.csv", sep=",", index=False)
        print("current_ok: ", current_ok)
        
