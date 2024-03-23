import os
import json

for filename in os.listdir("./results"):

    print()
    print(filename)

    f_in = open(f"./results/{filename}")
    data = json.load(f_in)
    f_in.close()

    bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"]

    row = " & ".join(bad_corpus_names) + " & Avg."
    print(row)

    values = [data[d]["accuracy"] for d in bad_corpus_names]
    avg = sum(values) / len(values)

    row = " & ".join(["{:.2f}".format(v) for v in values]) + " & " + "{:.2f}".format(avg)
    print(row)
