import os
import random

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

results = {}

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

choices = ["A","B","C","D"]

def getScoreRandom(seed):

    for filename in os.listdir("./csv+transcripts+dev/"):

        if "_test" not in filename or ".csv" not in filename:
            continue
        
        # print(filename)

        new_file_name = filename
        for mmlu_c in bad_corpus_names:
            new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
        splitted = new_file_name.replace(".csv","").split("_")
        corpus_name = splitted[0]
        subset_name = splitted[1]

        df = pd.read_csv(f"./csv+transcripts+dev/{filename}")

        refs = []
        preds = []

        for index, row in df.iterrows():
            refs.append(row["class"].upper())
            random.seed(seed)
            preds.append(random.choice(choices))

        acc = accuracy_score(refs, preds)
        # print(acc)

        results[corpus_name] = acc

    # print(results)

    arr = [results[corpus_name]*100 for corpus_name in good_corpus_name]
    avg = (sum(arr) / len(arr))

    row = " & ".join(["{:.1f}".format(results[corpus_name]*100) for corpus_name in good_corpus_name]) + " & " + "{:.1f}".format(avg)
    print(row)

getScoreRandom(1)
