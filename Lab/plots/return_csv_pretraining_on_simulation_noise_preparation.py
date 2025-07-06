import os
import pandas as pd
import numpy as np




df = pd.read_csv('./return_pretraining_virtual_testbed/1722503801.csv')
df["timestamp"] = 1722503801
df["goal_power_alg"] = "0.9"
df["goal"] = 0.9
df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
df["std1"] = df["Value"].ewm(alpha=0.01).std()
df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
df["Step"] = df["Step"]-500000
df.to_csv("return_training_pretrained_virtual_noise.csv")

def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df.groupby(['Step', 'goal_power_alg', 'goal']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]
stats_df["max_possible_return"] = 50*(np.exp(-1/6)+np.exp(0.93/stats_df["goal"]))
stats_df["mean_norm"] = stats_df["mean"]/stats_df["max_possible_return"]
stats_df["y_upper_norm"] = stats_df["y_upper"]/stats_df["max_possible_return"]
stats_df["y_lower_norm"] = stats_df["y_lower"]/stats_df["max_possible_return"]
stats_df["agent"] = "pretrained on virtual testbed with noise"
stats_df.to_csv("return_training_pretrained_virtual_noise_stats.csv")
print(stats_df)