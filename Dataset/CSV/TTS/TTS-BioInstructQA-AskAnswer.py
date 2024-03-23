import json
from openai import OpenAI

VOICES = ["alloy","echo","fable","onyx","nova","shimmer"]

LETTERS = ["A","B","C","D"]
ASK_OPTIONS = ["The correct answer is option "] + [f"The correct answer is option {letter.upper()}" for letter in LETTERS]

variants = {}

identifier = 0

client = OpenAI(api_key="sk-XXX")

for voice in VOICES:

    print(voice)
    
    for ask_option in ASK_OPTIONS:

        print(ask_option)

        output_path = f"./audios_ask/{identifier}.mp3"

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=ask_option,
        )

        response.stream_to_file(output_path)

        variants[identifier] = {
            "voice": voice,
            "ask_option": ask_option,
            "path": output_path,
            "path_wav": output_path.replace("./audios_ask/","./audios_ask_wav_ffmpeg/").replace(".mp3",".wav"),
        }
        # print(variants[identifier])

        identifier += 1

with open(f"./variants.json", 'w') as f:
    json.dump(variants, f, indent=4)
