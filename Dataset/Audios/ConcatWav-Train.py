import os
import json
import shutil
import argparse

import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from pqdm.processes import pqdm

parser = argparse.ArgumentParser(description="TTS medical question answering corpus")
parser.add_argument(
    "--nbr_threads",
    type=int,
    default=10,
    help="Number of threads.",
)
args = parser.parse_args()
assert args.nbr_threads != None

f_variants = open("./variants.json","r")
variants = json.load(f_variants)
variants = {(variants[v]["voice"], variants[v]["ask_option"][-1]): variants[v]["path_wav"] for v in variants}
f_variants.close()
print("variants:")
print(variants)

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def process(tuple_data):

    for t in tqdm(tuple_data):

        audio_path, speaker, correct_answer = t

        path_og = audio_path.replace("./audios/","./audios_wav_ffmpeg_train/")

        new_path_audio = audio_path.replace("./audios/","./audios_wav_ffmpeg_train_concat/")

        # sampling_rate = 16000

        # ask_audio_path = variants[(speaker, " ")]
        ask_audio_path = variants[(speaker, correct_answer)]

        os.system(
            # f"ffmpeg -y -i {path_og} -i {ask_audio_path} -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' {new_path_audio}"
            f"ffmpeg -y -i {path_og} -i {ask_audio_path} -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' {new_path_audio} > /dev/null 2>&1"
        )

for file_name in os.listdir("./json_train"):

    if has_numbers(file_name):
        continue
    
    print(file_name)
    
    f_in = open(f"./json_train/{file_name}","r")
    data = json.load(f_in)
    f_in.close()

    new_file_name = file_name
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    
    splitted = new_file_name.split("_")
    corpus_name = splitted[0].replace("-","_")
    subset = splitted[1].replace(".json","")

    if subset != "train":
        continue

    paths = [[d["audio_path"], d["speaker"], d["answer"]] for d in data]

    threads_paths = np.array_split(paths, args.nbr_threads)

    configs = []
    for i in range(0, args.nbr_threads):
        configs.append([threads_paths[i].tolist()])
    
    outputs = pqdm(configs, process, n_jobs=args.nbr_threads, argument_type='args')
    print(outputs)
