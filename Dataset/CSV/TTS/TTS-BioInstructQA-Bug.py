import json
import os.path

from tqdm import tqdm
from openai import OpenAI

f_in = open(f"./json/MMLU_college_medicine_test.json","r")
data = json.load(f_in)
f_in.close()

client = OpenAI(api_key="sk-XXX")

for d in tqdm(data):
    
    if os.path.isfile(d["audio_path"]) == True:
        continue

    print("Process: ", d["identifier"])

    current_voice = d["speaker"]

    print(len(d["prompt_no_answer"]))

    response = client.audio.speech.create(
        model="tts-1",
        voice=current_voice,
        input=d["prompt_no_answer"][4096:],
    )

    response.stream_to_file(d["audio_path"])

    print(d["audio_path"])
    print()
