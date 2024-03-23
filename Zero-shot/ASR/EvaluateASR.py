import re
import os
import json
import pandas as pd

import jiwer
from OfficialWhisperNormalizationEnglish import EnglishTextNormalizer

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

all_results = {
    "small": {},
    "medium": {},
    "large-v2": {},
}

whisper_normalizer = EnglishTextNormalizer()

for filename in os.listdir("./csv+transcripts+dev"):

    if "_test" not in filename:
        continue
    
    new_file_name = filename
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    splitted = new_file_name.replace(".csv","").split("_")
    print(splitted)

    corpus_name = splitted[0]
    subset = splitted[1]

    file_path = f"./csv+transcripts+dev/{filename}"
    
    df = pd.read_csv(file_path)

    local_data = {
        "references": [],
        "small": [],
        "medium": [],
        "large-v2": [],
    }
        
    for index, row in df.iterrows():

        local_data["references"].append(whisper_normalizer(row["wrd"]))

        local_data["small"].append(whisper_normalizer(row["small"]))
        local_data["medium"].append(whisper_normalizer(row["medium"]))
        local_data["large-v2"].append(whisper_normalizer(row["large-v2"]))

    all_results["small"][corpus_name] = jiwer.wer(local_data["references"], local_data["small"], truth_transform=None)*100
    all_results["medium"][corpus_name] = jiwer.wer(local_data["references"], local_data["medium"], truth_transform=None)*100
    all_results["large-v2"][corpus_name] = jiwer.wer(local_data["references"], local_data["large-v2"], truth_transform=None)*100

print(json.dumps(all_results, sort_keys=True, indent=4))

row = " & " + " & ".join(["small","medium","large-v2"])
print(row)

for corpus_name in good_corpus_name:

    row = corpus_name + " & " + " & ".join(["{:.2f}".format(all_results[whisper_model][corpus_name]) for whisper_model in ["small","medium","large-v2"]])
    print(row)
        
        
