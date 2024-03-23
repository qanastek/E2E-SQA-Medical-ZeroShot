import json
import argparse

from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

# python Predict.py --batch_size=32 --model_name="/users/ylabrak/SpokenMedicalQA/SpeechClassification/models/whisper-small-finetuned-MedMCQA"

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
args = parser.parse_args()
args = vars(args)

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

full_name = args["model_name"].split("/")[-1]

batch_size = args["batch_size"]
# batch_size = 32
# batch_size = 16

pipe = pipeline(
    "audio-classification",
    model=args["model_name"],
    device=0
)

def chunks(lst, n):
    res_chunk = []
    for i in range(0, len(lst), n):
        res_chunk.append(lst[i:i + n])
    return res_chunk

all_results = {}

for corpus_name in bad_corpus_names:

    dataset = load_dataset("SpeechLLM/SpokenMedicalQA", corpus_name)["test"]

    references = []
    hypothesis = []

    paths = [(d["audio"]["path"], d["class"]) for d in dataset]
    batches = chunks(paths, batch_size)

    for batch in tqdm(batches):

        batch_paths  = [b[0] for b in batch]
        batch_labels = [b[1] for b in batch]

        batch_res = pipe(batch_paths)

        for res_preds, res_labels in zip(batch_res, batch_labels):

            scores = [r["score"] for r in res_preds]
            labels = [r["label"] for r in res_preds]

            max_value = max(scores)
            max_index = scores.index(max_value)
            predicted_label = labels[max_index]

            references.append(res_labels)
            hypothesis.append(predicted_label)

    acc = accuracy_score(references, hypothesis)
    print(acc*100)

    all_results[corpus_name] = {
        "accuracy": acc*100,
        "references": references,
        "hypothesis": hypothesis,
    }

with open(f"./results/{full_name}.json", 'w') as f:
    json.dump(all_results, f)
