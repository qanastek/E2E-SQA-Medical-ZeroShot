import time
import json
import os.path
import argparse

import numpy as np
from tqdm import tqdm
from openai import OpenAI
from pqdm.processes import pqdm
from datasets import load_dataset

parser = argparse.ArgumentParser(description="TTS medical question answering corpus")

parser.add_argument(
    "--nbr_threads",
    type=int,
    default=7,
    help="Number of threads.",
)
args = parser.parse_args()
assert args.nbr_threads != None

IDX = 0
VOICES = ["alloy","echo","fable","onyx","nova","shimmer"]

def switch_voice():
    global IDX

    v = VOICES[IDX]

    if IDX < len(VOICES)-1:
        IDX += 1
    elif IDX >= len(VOICES)-1:
        IDX = 0
    
    return v

all_corpus = ["MedQA","MedMCQA"]
all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

def process(corpus_n, subset_n, nbr_threads, thread_index):


    dataset = load_dataset("SpeechLLM/BioInstructQA", corpus_n)

    subset_identifiers = [d["identifier"] for d in dataset[subset_n]]

    # Split the remaining data into equal sized chunks
    all_chunks = np.array_split(subset_identifiers, nbr_threads)

    # Collect the chunk of data to process for the current thread
    thread_ids = all_chunks[thread_index].tolist()

    client = OpenAI(api_key="sk-XXX")

    current_thread_path = f"./json/{corpus_n}_{subset_n}_{thread_index}.json"

    if os.path.isfile(current_thread_path):
        fc_thread = open(current_thread_path,"r")
        all_data = json.load(fc_thread)
        fc_thread.close()
    else:
        all_data = []
    
    ids_already_done = [ad["identifier"] for ad in all_data]

    cpt = 0

    for d in tqdm(dataset[subset_n]):

        audio_path = f"./audios/{corpus_n}_{subset_n}_{d['identifier']}.wav"

        if d["identifier"] not in thread_ids:
            continue
        if d["identifier"] in ids_already_done and os.path.isfile(audio_path):
            continue

        cpt += 1

        current_voice = switch_voice()

        d["speaker"] = current_voice
        d["audio_path"] = audio_path
        all_data.append(d)
        
        try:

            time.sleep(3)

            response = client.audio.speech.create(
                model="tts-1",
                voice=current_voice,
                input=d["prompt_no_answer"],
            )

            response.stream_to_file(audio_path)
        
        except Exception as e:
            print(e)

        if cpt % 10 == 0:
            with open(current_thread_path, 'w') as f:
                json.dump(all_data, f, indent=4)

    with open(current_thread_path, 'w') as f:
        json.dump(all_data, f, indent=4)
        

for corpus_name in all_corpus:

    print(corpus_name)

    for subset in ["test"]:
        
        # Load all the previous thread data
        already_dones = [pa for pa in os.listdir("./audios") if f"{corpus_name}_{subset}" in pa]
        print("Already done: ", len(already_dones))
                
        # Create all the configs for the threads
        configs = []
        for i in range(0, args.nbr_threads):
            configs.append([corpus_name, subset, args.nbr_threads, i])
        
        # process(*configs[0])
        # print("#"*50)
        outputs = pqdm(configs, process, n_jobs=args.nbr_threads, argument_type='args')
        print(outputs)

        threads_merged_data = []

        for id_t in range(args.nbr_threads):
            f_thread = open(f"./json/{corpus_name}_{subset}_{id_t}.json","r")
            data_t = json.load(f_thread)
            f_thread.close()
            threads_merged_data.extend(data_t)

        ds = load_dataset("SpeechLLM/BioInstructQA", corpus_name)[subset]
        print(corpus_name)
        print("Total size: ", len(threads_merged_data), " out of ", len(ds))

        with open(f"./json/{corpus_name}_{subset}.json", 'w') as f:
            json.dump(threads_merged_data, f, indent=4)
