import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv("different_hand_mirror_pos_test.csv", index_col=0)
df["agent"] = "pretrained_lower_goals_test_at_diff_pos"

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
df_statistics.to_csv("diff_hand_mirror_pos_stats.csv")
print(df_statistics.drop('agent', axis=1).mean())
df_only_first_episode = df.groupby('reset_number').filter(lambda x: ((x['reset_number'] == 0).any()))
df_power_after_first_episode = df_only_first_episode[["try", "Power"]].groupby(["try"]).agg("last").reset_index(drop=False)
df_only_first_episode_goal = df_only_first_episode.groupby(['agent', "try", 'reset_number']).filter(lambda x: ((x['Power'].iloc[-1] >= 0.9)))
df_episode_lengths_first_episode = df_only_first_episode_goal.groupby(['agent', "try", 'reset_number']).size().reset_index(name='steps_to_reach_goal')
df_episode_lengths_first_episode_mean = df_episode_lengths_first_episode["steps_to_reach_goal"].agg("mean")
print(df_episode_lengths_first_episode_mean)

