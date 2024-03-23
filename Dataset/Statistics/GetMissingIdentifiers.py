import os
import json

from datasets import load_dataset

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

for file_name in os.listdir("./json"):

    if has_numbers(file_name):
        continue
    
    print("*"*50)
    print(file_name)
    
    f_in = open(f"./json/{file_name}","r")
    data = json.load(f_in)
    f_in.close()

    ids_data_list = [d["identifier"] for d in data]
    print(len(ids_data_list))
    print("-"*10)

    ids_data = set(ids_data_list)
    print(len(ids_data))
    print("-"*10)

    diff = [d for d in list(ids_data) if d not in ids_data_list]
    print(diff)

    print(len(ids_data_list) == len(ids_data))
    print("-"*10)

    new_file_name = file_name
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    
    splitted = new_file_name.split("_")
    corpus_name = splitted[0].replace("-","_")
    subset = splitted[1].replace(".json","")

    ds = load_dataset("SpeechLLM/BioInstructQA", corpus_name)[subset]
    ids_ds = [d["identifier"] for d in ds]

    for id_d in ids_ds:

        if id_d not in ids_data:
            print(id_d)
