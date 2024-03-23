import argparse
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoModelForAudioClassification

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
args = parser.parse_args()
args = vars(args)

# python FineTuningWav2Vec.py --model_name="openai/whisper-small" --batch_size=32
# python FineTuningWav2Vec.py --model_name="openai/whisper-medium" --batch_size=12
# python FineTuningWav2Vec.py --model_name="openai/whisper-large-v2" --batch_size=6

# python FineTuningWav2Vec.py --model_name="microsoft/wavlm-large" --batch_size=32

dataset = load_dataset("SpeechLLM/SpokenMedicalQATrain","MedMCQA")

model_id = args["model_name"]
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=False
    # return_attention_mask=True
)
feature_extractor.truncation_side = "left"

max_duration = 30.0
sampling_rate = feature_extractor.sampling_rate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=False,
        # return_attention_mask=True,
    )
    return inputs

dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns=["ID_original_dataset", "ID", "duration", "wav", "spk_id", "wrd", "small", "medium", "large-v2", "audio"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
print(dataset_encoded)

# dataset_encoded = dataset_encoded.rename_column("class", "label")

id2label_fn = dataset["train"].features["label"].int2str
id2label = {
    str(i): id2label_fn(i) for i in range(len(dataset_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

model_name = model_id.split("/")[-1]
batch_size = args["batch_size"]
# batch_size = 32
# batch_size = 24
# batch_size = 12
gradient_accumulation_steps = 1
num_train_epochs = 3

training_args = TrainingArguments(
    f"{model_name}-finetuned-MedMCQA",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(f"./models/{model_name}-finetuned-MedMCQA")
