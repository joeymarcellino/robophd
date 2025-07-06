import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from statistics import mean
matplotlib.use('TkAgg')

df0 = pd.read_csv("pretrained_on_virtual/1715764610_500000_goal0.9_fail0.05_start0.2.csv",
                  index_col=0)
df0["agent"] = "only trained on virtual"
df1 = pd.read_csv("pretrained_on_virtual/1715850724_218000_goal0.9_fail0.05_start0.2.csv",
                  index_col=0)
df1["agent"] = "pretrained on virtual testbed"
df2 = pd.read_csv("1714375298_204000_goal0.9_fail0.05_start0.2.csv", index_col=0)
df2["agent"] = "pretrain on lower goals"

df = pd.concat([df0, df1], ignore_index=True)
df = pd.concat([df, df2], ignore_index=True)
print(df)
def last_value(group):
    """
    for index, row in group.iloc[::-1].iterrows():
        if row['time_in_s'] <= 40:  # Check if the value in column2 exceeds the threshold
            return row['Power']
    return None
    """
    for index, row in group.iloc[::-1].iterrows():
        return row['Power']

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
episode_lengths = df.groupby(['agent', "try", 'reset_number']).size().reset_index(name='steps_last_episode')
episode_lengths['steps_last_episode'] = episode_lengths['steps_last_episode'] - 1
df_starting_power = df[['agent', "try", "Power"]].groupby(['agent', "try"]).agg("first").reset_index(drop=False)
print(df_starting_power)
# Find the last episode for each agent
last_episodes = episode_lengths.groupby(['agent', "try"])['reset_number'].max().reset_index()


# Merge to get the length of the last episode for each agent
df_steps_last_episode = pd.merge(last_episodes, episode_lengths, on=['agent', "try", 'reset_number'])
df_statistics = pd.merge(df_end_power, df_max_time)
df_statistics = pd.merge(df_statistics, df_steps_last_episode)
df_statistics = pd.merge(df_statistics, df_starting_power)

print(df_statistics)
df_statistics.to_csv("pretraining_stats.csv")
df_statistics = df_statistics.groupby('agent').filter(lambda x: ((x['agent'] == "pretrain on lower goals").any()))
print(df_statistics)
print(df_statistics.groupby(["agent"]).mean())

df_only_first_episode = df.groupby('reset_number').filter(lambda x: ((x['reset_number'] == 0).any()))
df_power_after_first_episode = df_only_first_episode[['agent', "try", "Power"]].groupby(['agent', "try"]).agg("last").reset_index(drop=False)
df_only_first_episode_goal = df_only_first_episode.groupby(['agent', "try", 'reset_number']).filter(lambda x: ((x['Power'].iloc[-1] >= 0.9)))
df_episode_lengths_first_episode = df_only_first_episode_goal.groupby(['agent', "try", 'reset_number']).size().reset_index(name='steps_to_reach_goal')
print(df_episode_lengths_first_episode)

def probability_reaching_goal(group):
    prob = 0.0
    for i in range(100):
        if group['Power'].iloc[i] >= 0.9:
            prob += 0.01
    return prob

def probability_failing(group):
    prob = 0.0
    for i in range(100):
        if group['Power'].iloc[i] < 0.05:
            prob += 0.01
    return prob

df_prob_goal_first_episode = df_power_after_first_episode.groupby(['agent']).apply(probability_reaching_goal).reset_index(name="prob_goal")
df_prob_fail_first_episode = df_power_after_first_episode.groupby(['agent']).apply(probability_failing).reset_index(name="prob_fail")
df_statistics_first_episode = pd.merge(df_prob_goal_first_episode, df_prob_fail_first_episode)
df_statistics_first_episode["prob_neither"] = 1 - df_statistics_first_episode["prob_goal"] - df_statistics_first_episode["prob_fail"]
df_episode_lengths_first_episode_mean = df_episode_lengths_first_episode.groupby(["agent"]).agg("mean").reset_index(drop=False)
df_statistics_first_episode = pd.merge(df_statistics_first_episode, df_episode_lengths_first_episode_mean)

print(df_statistics_first_episode)
df_statistics_first_episode.to_csv("pretraining_stats_first_episode.csv")