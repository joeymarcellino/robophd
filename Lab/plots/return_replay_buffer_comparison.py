import os
import pandas as pd
import numpy as np



files = os.listdir('./return_pretraining_on_lower_goal/') #insert folder path
df_total = pd.DataFrame(data=None, index=None)
df_without_replay = pd.DataFrame(data=None, index=None)
for file in files:
    df = pd.read_csv('./return_pretraining_on_lower_goal/'+file)
    file = file[:-4]
    goal_power, timestamp = file.split('_')
    df["smoothed_value"] = df["Value"].ewm(alpha=0.01).mean()
    df["std1"] = df["Value"].ewm(alpha=0.01).std()
    df["smoothed_value_squared_plus_std1_squared"] = (df["smoothed_value"].apply(lambda x: x**2)
                                                      + df["std1"].apply(lambda x: x**2))
    df["timestamp"] = timestamp
    if (timestamp == "1714246042" or timestamp == "1714485701"
            or timestamp == "1715453496"):
        df["goal_power_replay"] = goal_power + "_with_replay"
        print(goal_power + "_with_replay")
    elif timestamp == "1714153135":
        df["goal_power_replay"] = goal_power
        df_without_replay = pd.concat([df_without_replay, df], ignore_index=True)
    else:
        df["goal_power_replay"] = goal_power+"_without_replay"
        print(goal_power+"_without_replay")
        df_without_replay = pd.concat([df_without_replay, df], ignore_index=True)
    df["goal_power_timestamp"] = goal_power+"_"+timestamp
    df_total = pd.concat([df_total, df], ignore_index=True)
print(df_total)
df_total.to_csv("return_replay_buffer_tests.csv")

def std_combined(group):
    mean_means = group["smoothed_value"].mean()
    std = np.sqrt(group["smoothed_value_squared_plus_std1_squared"].sum()/(group["smoothed_value"].size)
                  - mean_means**2)
    return pd.Series({"mean": mean_means, "std": std})

stats_df = df_total.groupby(['Step', 'goal_power_replay']).apply(std_combined).reset_index()

stats_df["y_upper"] = stats_df["mean"]+2*stats_df["std"]
stats_df["y_lower"] = stats_df["mean"]-2*stats_df["std"]

stats_df.to_csv("return_replay_buffer_stats.csv")
print(stats_df)
stats_df_without_replay = df_without_replay.groupby(['Step', 'goal_power_replay']).apply(std_combined).reset_index()

stats_df_without_replay["y_upper"] = stats_df_without_replay["mean"]+2*stats_df_without_replay["std"]
stats_df_without_replay["y_lower"] = stats_df_without_replay["mean"]-2*stats_df_without_replay["std"]
stats_df_without_replay["agent"] = "pretrain on lower goals"
stats_df_without_replay.to_csv("return_replay_buffer_stats_without_replay.csv")