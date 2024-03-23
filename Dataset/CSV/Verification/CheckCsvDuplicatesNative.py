import os
import csv

for file_path in os.listdir("./csv"):

    if ".csv" not in file_path:
        continue

    print(file_path)
    csv_reader = csv.reader(f"./csv/{file_path}", delimiter=",")
    
    ids = []
    for row in csv_reader:
        ids.append(row[0])
    print(len(ids))

    ids_deduplicated = list(set(ids))
    print(len(ids_deduplicated))

    diff = len(ids_deduplicated) == len(ids)
    print(diff)
    
    print()
