import os
import pandas as pd
import numpy as np



files = os.listdir('./return_csv_training_long/') #insert folder path
df_total = pd.DataFrame(data=None, index=None)
for file in files:
    df = pd.read_csv('./return_csv_training_long/'+file)
    goal_power, timestamp, algorithm = file[:-4].split('_')
    print(goal_power, timestamp, algorithm)
    df["goal"] = float(goal_power)
    df["timestamp"] = timestamp
    if algorithm == "TQCpretrained":
        algorithm = "TQC_pretrained"
    df["goal_power_alg"] = goal_power+"_"+algorithm
    df["goal_power_timestamp"] = goal_power+"_"+file[-14:-4]
    df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
    df["std1"] = df["Value"].ewm(alpha=0.01).std()
    df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
    df_total = pd.concat([df_total, df], ignore_index=True)
print(df_total)
df_total.to_csv("return_trainings_long.csv")

def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df_total.groupby(['Step', 'goal_power_alg', 'goal']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]
stats_df["max_possible_return"] = 50*(np.exp(-1/6)+np.exp(0.93/stats_df["goal"]))
stats_df["mean_norm"] = stats_df["mean"]/stats_df["max_possible_return"]
stats_df["y_upper_norm"] = stats_df["y_upper"]/stats_df["max_possible_return"]
stats_df["y_lower_norm"] = stats_df["y_lower"]/stats_df["max_possible_return"]

stats_df.to_csv("return_trainings_long_stats.csv")
print(stats_df)