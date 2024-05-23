import os
import pandas as pd
import numpy as np




df = pd.read_csv('./return_pretraining_virtual_testbed/1715850724.csv')
df["timestamp"] = 1715850724
df["goal_power_alg"] = "0.9"
df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
df["std1"] = df["Value"].ewm(alpha=0.01).std()
df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
df["Step"] = df["Step"]-500000
df.to_csv("return_training_pretrained_vitual.csv")

def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df.groupby(['Step', 'goal_power_alg']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]
stats_df["agent"] = "pretrained on virtual testbed"
stats_df.to_csv("return_training_pretrained_virtual_stats.csv")
print(stats_df)