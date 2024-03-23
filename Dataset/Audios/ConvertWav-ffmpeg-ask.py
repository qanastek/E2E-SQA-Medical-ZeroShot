import os

for path_mp3 in os.listdir("./audios_ask"):

    path_mp3 = f"./audios_ask/{path_mp3}"
    
    print(path_mp3)

    new_path_audio = path_mp3.replace("./audios_ask/","./audios_ask_wav_ffmpeg/").replace(".mp3",".wav")

    sampling_rate = 16000

    os.system(
        f"ffmpeg -y -i {path_mp3} -ac 1 -ar {sampling_rate} {new_path_audio} > /dev/null 2>&1"
    )
