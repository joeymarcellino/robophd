import os
import pandas as pd
import numpy as np

files = os.listdir('./return_start_from_0/') #insert folder path
files.remove("0.88_1714976689.csv")
files.remove("0.86_1714884937.csv")
print(files)
df_total = pd.DataFrame(data=None, index=None)
for file in files:
    df = pd.read_csv('./return_start_from_0/'+file)
    file = file[:-4]
    goal_power, timestamp = file.split('_')
    if goal_power == "0.88":
        df88_2 = pd.read_csv('./return_start_from_0/0.88_1714976689.csv')
        df = pd.concat([df, df88_2], ignore_index=True)
    if goal_power == "0.86":
        df86_2 = pd.read_csv('./return_start_from_0/0.86_1714884937.csv')
        df = pd.concat([df, df86_2], ignore_index=True)
    df["goal_power"] = goal_power
    df["goal"] = float(goal_power)
    df["timestamp"] = timestamp
    df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
    df["std1"] = df["Value"].ewm(alpha=0.01).std()
    df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
    df_total = pd.concat([df_total, df], ignore_index=True)
print(df_total)
df_total.to_csv("return_start_from_0.csv")

def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df_total.groupby(['Step', 'goal_power', 'goal']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]
stats_df["max_possible_return"] = 50*(np.exp(-1/6)+np.exp(0.93/stats_df["goal"]))
stats_df["mean_norm"] = stats_df["mean"]/stats_df["max_possible_return"]
stats_df["y_upper_norm"] = stats_df["y_upper"]/stats_df["max_possible_return"]
stats_df["y_lower_norm"] = stats_df["y_lower"]/stats_df["max_possible_return"]

stats_df.to_csv("return_start_from_0_stats.csv")
print(stats_df)