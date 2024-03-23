import os
import pandas as pd


for file_path in os.listdir("./csv"):

    if ".csv" not in file_path:
        continue

    print(file_path)
    df = pd.read_csv(f"./csv/{file_path}", sep=",")
    
    ids = []
    for index, row in df.iterrows():
        ids.append(row['ID'])
    print(len(ids))

    ids_deduplicated = list(set(ids))
    print(len(ids_deduplicated))

    diff = len(ids_deduplicated) == len(ids)
    print(diff)
    
    print()
