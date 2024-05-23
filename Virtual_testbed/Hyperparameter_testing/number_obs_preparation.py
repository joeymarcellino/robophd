import os
import pandas as pd
import numpy as np

files = os.listdir('./number_obs/') #insert folder path
df_total = pd.DataFrame(data=None, index=None)
for file in files:
    df = pd.read_csv('./number_obs/'+file)
    file = file[:-10]
    file = file[10:]
    number_obs, timestamp = file.split('_')
    df["number_obs"] = number_obs
    df["timestamp"] = timestamp
    df["rounded_step"] = df["Step"]/5
    df = df.round({"rounded_step": -2})
    df["rounded_step"] = df["rounded_step"]*5
    df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
    df["std1"] = df["Value"].ewm(alpha=0.01).std()
    df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
    df_total = pd.concat([df_total, df], ignore_index=True)
print(df_total)
df_total.to_csv("number_obs_tests.csv")
def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df_total.groupby(['rounded_step', 'number_obs']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]


stats_df.to_csv("number_obs_stats.csv")
print(stats_df)