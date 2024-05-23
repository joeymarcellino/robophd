import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

df0 = pd.read_csv("pretrained_on_virtual/1715764610_500000.csv",
                  index_col=0)
df0["agent"] = "only trained on virtual"
df1 = pd.read_csv("pretrained_on_virtual/1715850724_218000.csv",
                  index_col=0)
df1["agent"] = "pretrained on virtual testbed"
df2 = pd.read_csv("1714375298_204000.csv", index_col=0)
df2["agent"] = "pretrain on lower goals"

df = pd.concat([df0, df1], ignore_index=True)
df = pd.concat([df, df2], ignore_index=True)
print(df)
def last_value(group):
    for index, row in group.iloc[::-1].iterrows():
        if row['time_in_s'] <= 40:  # Check if the value in column2 exceeds the threshold
            return row['Power']
    return None
df_end_power = df.groupby(['agent', 'try']).apply(last_value).reset_index(drop=False)
df_end_power.columns = ["agent", "try", "Power_end"]
print(df_end_power)
def trial_length(group):
    if group['Power'].iloc[-1] >= 0.9:
        return group['time_in_s'].max()
    else:
        return None
df_max_time = df.groupby(['agent', 'try']).apply(trial_length).reset_index(drop=False)
df_max_time.columns = ["agent", "try", "max_time"]
df_statistics = pd.merge(df_end_power, df_max_time)
print(df_statistics)
df_statistics.to_csv("pretraining_stats.csv")
