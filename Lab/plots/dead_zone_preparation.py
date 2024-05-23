import os
import pandas as pd

files = os.listdir('./dead_zone/') #insert folder path
df_total = pd.DataFrame(data=None, index=None)
for file in files:
    df = pd.read_csv('./dead_zone/'+file)
    print(file[14:-4])
    df["actuator"] = file[14:-4]
    df_total = pd.concat([df_total, df], ignore_index=True)
print(df_total)
df_total.to_csv("backlash_study.csv")