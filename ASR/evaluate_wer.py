import speechbrain
import csv

def load_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

file_path = "/workspace/SpokenMedicalQA_data/data-30+/MedMCQA_test.csv"


data = load_csv(file_path)
header = data[0]
error_rate_computer = speechbrain.utils.metric_stats.ErrorRateStats()

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v2")
normalize = True 

for i in range(len(data)):
    if i == 0:
        continue 

    if normalize:
        t_words = tokenizer._normalize(data[i][-5])
        p_words = tokenizer._normalize(data[i][-2])
    else:
        t_words = data[i][-5]
        p_words = data[i][-2]

    target_words = [t_words.split(" ")]
    predicted_words = [p_words.split(" ")]
    ids = [i] 

    error_rate_computer.append(ids, predicted_words, target_words)


with open(f"test_normalized={normalize}.txt", "w") as w:
    error_rate_computer.write_stats(w)