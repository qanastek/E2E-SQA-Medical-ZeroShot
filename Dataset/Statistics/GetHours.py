import os
import datetime

import pandas as pd

for filename in os.listdir("./csv+transcripts+dev/"):

    if ".csv" not in filename:
        continue
    
    print()
    print(filename)

    total = 0.0

    df = pd.read_csv(f"./csv+transcripts+dev/{filename}")

    for index, row in df.iterrows():
        total += row["duration_no_answer"]
    
    print(total)
    print(str(datetime.timedelta(seconds=total)))

