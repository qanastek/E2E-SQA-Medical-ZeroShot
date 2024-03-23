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

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def process(audio_paths):

    for audio_path in tqdm(audio_paths):

        if os.path.isfile(audio_path) == False:
            print("Error: No file found! - ", audio_path)
            print(new_path_audio)
            print()
            continue

        new_path_audio = audio_path.replace("./audios/","./audios_wav/")
        
        sound = AudioSegment.from_mp3(audio_path)
        sound.export(new_path_audio, format="wav", bitrate="16k")

for file_name in ["MMLU_college_medicine_test.json"]:

    if has_numbers(file_name):
        continue
    
    print(file_name)
    
    f_in = open(f"./json/{file_name}","r")
    data = json.load(f_in)
    f_in.close()

    new_file_name = file_name
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    
    splitted = new_file_name.split("_")
    corpus_name = splitted[0].replace("-","_")
    subset = splitted[1].replace(".json","")

    if subset != "test":
        continue

    paths = [d["audio_path"] for d in data]

    process(paths)
