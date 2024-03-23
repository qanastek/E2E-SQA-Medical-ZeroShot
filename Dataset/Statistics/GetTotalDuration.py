import os
import librosa
import soundfile as sf

from tqdm import tqdm

# Test : 44,30 hours

total_duration = []

for file_path in tqdm(os.listdir("./audios_wav")):
    f = sf.SoundFile(f"./audios_wav/{file_path}")
    res = f.frames / f.samplerate
    total_duration.append(res)

print(sum(total_duration), " seconds")
