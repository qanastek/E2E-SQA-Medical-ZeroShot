import os

from collections import Counter

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]

subsets = {
    "train": [],
    "test": [],
}
expected = {
    "train": {'MedMCQA': 146257, 'MedQA': 10178},
    "test": {'MedMCQA': 4183, 'MedQA': 1273, 'MMLU_professional_medicine': 272, 'MMLU_clinical_knowledge': 265, 'MMLU_college_medicine': 173, 'MMLU_college_biology': 144, 'MMLU_anatomy': 135, 'MMLU_medical_genetics': 100},
}

for file_path in os.listdir("./audios_mp3"):

    new_file_name = file_path
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    
    splitted = new_file_name.split("_")
    corpus_name = splitted[0].replace("-","_")
    subset = splitted[1].replace(".json","")

    subsets[subset].append(corpus_name)

for subset in subsets:

    print(subset)
    ct = Counter(subsets[subset])
    print(ct)
    print("-"*50)
    print("expected: ", expected[subset])
    print()
