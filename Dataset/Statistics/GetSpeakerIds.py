import json
from collections import Counter
import matplotlib.pyplot as plt

bad_corpus_names = {
    "test" : ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"],
    "train" : ["MedQA","MedMCQA"]
}

for _subset in ["test","train"]:

    print(_subset)
    
    for ALLOWED_CORPUS in bad_corpus_names[_subset]:
        
        print(ALLOWED_CORPUS)

        all_speakers = []

        for i in range(7):

            path_file = f"./json/{ALLOWED_CORPUS}_{_subset}_{i}.json"
                
            f_in = open(path_file,"r")
            data = json.load(f_in)
            f_in.close()

            speakers = [d["speaker"] for d in data]
            all_speakers.extend(speakers)
        
        # print(all_speakers)
        ct = Counter(all_speakers).most_common(9999)
        print(ct)

        speakers_names = [c[0] for c in ct]
        speakers_count = [c[1] for c in ct]
        
        # plt.hist(speakers_count)
        
        plt.bar(speakers_names, speakers_count, color ='maroon', width = 0.4)
        plt.savefig(f"./histograms_speakers/{ALLOWED_CORPUS}_{_subset}.png")
        plt.clf()
