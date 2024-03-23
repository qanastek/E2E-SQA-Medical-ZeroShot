import json
import os.path

from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key="sk-XXX")

IDX = 0
VOICES = ["alloy","echo","fable","onyx","nova","shimmer"]

def switch_voice():
    global IDX

    v = VOICES[IDX]

    if IDX < len(VOICES)-1:
        IDX += 1
    elif IDX >= len(VOICES)-1:
        IDX = 0
    
    return v

all_corpus = ["MedQA","MedMCQA"]
all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

for corpus_name in all_corpus:

    print(corpus_name)

    dataset = load_dataset("SpeechLLM/BioInstructQA",corpus_name)

    for subset in ["test"]:

        all_data = []

        cpt = 0

        for d in tqdm(dataset[subset]):

            cpt += 1

            audio_path = f"./audios/{corpus_name}_{subset}_{d['identifier']}.wav"

            current_voice = switch_voice()

            d["speaker"] = current_voice
            d["audio_path"] = audio_path
            all_data.append(d)

            if os.path.isfile(audio_path) == False:

                response = client.audio.speech.create(
                    model="tts-1",
                    voice=current_voice,
                    input=d["prompt_no_answer"],
                )

                response.stream_to_file(audio_path)

            if cpt % 100 == 0:
                with open(f"./json/{corpus_name}_{subset}.json", 'w') as f:
                    json.dump(all_data, f, indent=4)
        
        with open(f"./json/{corpus_name}_{subset}.json", 'w') as f:
            json.dump(all_data, f, indent=4)
