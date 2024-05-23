import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

df_human = pd.read_csv("./human/goal0.85_fail0.05_start0.2.csv", index_col=0)
df_human["agent"] = "human"
df_human["reset_number"] = 0
df_best_90 = pd.read_csv("comparison.csv", index_col=0)
df = pd.concat([df_human, df_best_90], ignore_index=True)
print(df)
def last_value(group):
    for index, row in group.iloc[::-1].iterrows():
        if row['time_in_s'] <= 40:  # Check if the value in column2 exceeds the threshold
            return row['Power']
    return None
df_end_power = df.groupby(['agent', 'try']).apply(last_value).reset_index(drop=False)
df_end_power.columns = ["agent", "try", "Power_end"]
def trial_length(group):
    return group['time_in_s'].max()

df_max_time = df.groupby(['agent', 'try']).apply(trial_length).reset_index(drop=False)
df_max_time.columns = ["agent", "try", "max_time"]

episode_lengths = df.groupby(['agent', "try", 'reset_number']).size().reset_index(name='steps_last_episode')
episode_lengths['steps_last_episode'] = episode_lengths['steps_last_episode'] - 1
# Find the last episode for each agent
last_episodes = episode_lengths.groupby(['agent', "try"])['reset_number'].max().reset_index()

# Merge to get the length of the last episode for each agent
df_steps_last_episode = pd.merge(last_episodes, episode_lengths, on=['agent', "try", 'reset_number'])


df_statistics = pd.merge(df_end_power, df_max_time)
df_statistics = pd.merge(df_statistics, df_steps_last_episode)
print(df_statistics)
df_statistics.to_csv("human_vs_agnes_statistics.csv")
