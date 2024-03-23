import os
import json

import pandas as pd

DIR_PATH = "/mnt/d/Projects/LIA/SpokenMedicalQA/csv"

for filename in os.listdir(DIR_PATH):

    if "_test" not in filename:
        continue

    file_path = f"{DIR_PATH}/{filename}"

    df = pd.read_csv(file_path, sep=",")

    f_out = open(f"./jsonl/{filename.replace('.csv','.jsonl')}","w")

    for index, row in df.iterrows():

        item = {
            "audio_path": row['wav'],
            "category": f"[{row['class'].lower()}]"
        }

        f_out.write(json.dumps(item) + "\n")
    
    f_out.close()
