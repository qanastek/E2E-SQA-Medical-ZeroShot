import os
import datetime
import soundfile as sf

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Test : 44,30 hours

# Train MedMCQA : 15,65 heures
# Train MedQA : 20,76 heures

# ls -l | grep _train | grep MedMCQA | wc -l

bad_corpus_names = {
    "test" : ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"],
    "train" : ["MedQA","MedMCQA"]
}

for _subset in ["test","train"]:

    print("-"*50)
    
    for ALLOWED_CORPUS in bad_corpus_names[_subset]:
    # for ALLOWED_CORPUS in ["MedQA","MedMCQA"]:
        
        print(_subset, " - ", ALLOWED_CORPUS)
        print("-"*5)

        is_allowed = []
        total_duration = []
        total_trunked_duration = []

        for file_path in os.listdir("./audios"):
        # for file_path in tqdm(os.listdir("./audios_wav")):
            
            if _subset not in file_path or ALLOWED_CORPUS not in file_path:
                continue
            
            f = sf.SoundFile(f"./audios/{file_path}")
            res = f.frames / f.samplerate
            
            total_duration.append(res)
            is_allowed.append(res <= 30)
            if res > 30:
                total_trunked_duration.append(res-30)

        seconds = sum(total_duration)
        print(seconds, " seconds")
        print("Total time (minutes): ", str(seconds // 60))
        print("Total time (hours): ", str(seconds // 3600))
        td = datetime.timedelta(seconds=seconds)
        print("Total time: ", str(td))
        print("Total allowed", sum(is_allowed), "on the", len(is_allowed), "files.", "{:.2f}".format((sum(is_allowed)/len(is_allowed))*100), " %")

        print("="*10)
        total_removed = sum(total_trunked_duration)
        print("Total to remove (seconds): ", total_removed)        
        print("Total to remove (minutes): ", total_removed // 60)        
        print("Total to remove (hours): ", total_removed // 3600)        
        print("Total time left after removing (seconds): ", seconds-total_removed)
        td_r = datetime.timedelta(seconds=seconds-total_removed)
        print("Total time left after removing: ", str(td_r))
        print("Total to remove (%) : ", "{:.2f}".format((total_removed / seconds)*100), " %")
        print()
                
        plt.hist([int(t) for t in total_duration])
        plt.savefig(f"./histograms/{ALLOWED_CORPUS}_{_subset}.png")
        plt.clf()
