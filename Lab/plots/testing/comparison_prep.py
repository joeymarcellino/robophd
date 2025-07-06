import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

df1 = pd.read_csv(f"1715087568_169000_goal0.9_fail0.05_start0.2.csv", index_col=0)
df1["agent"] = "0.9_TQC"
df2 = pd.read_csv(f"1714375298_204000_goal0.9_fail0.05_start0.2.csv", index_col=0)
df2["agent"] = "0.9_TQC_pretrained"
df4 = pd.read_csv(f"1714915417_35000_goal0.87_fail0.05_start0.2.csv", index_col=0)
df4["agent"] = "0.87_TQC"
df5 = pd.read_csv(f"1714153135_37000_goal0.85_fail0.05_start0.2.csv", index_col=0)
df5["agent"] = "0.85_TQC"
df6 = pd.read_csv(f"1749908988_34000_goal0.85_fail0.05_start0.2.csv", index_col=0)
df6["agent"] = "0.85_CrossQ"
df7 = pd.read_csv(f"1750325230_113000_goal0.9_fail0.05_start0.2.csv", index_col=0)
df7["agent"] = "0.9_CrossQ"
df8 = pd.read_csv(f"1714656374_34000_goal0.85_fail0.05_start0.2.csv", index_col=0)
df8["agent"] = "0.85_SAC"
df9 = pd.read_csv(f"1750687596_122000_goal0.9_fail0.05_start0.2.csv", index_col=0)
df9["agent"] = "0.9_SAC"
df = pd.concat([df1, df2, df4, df5, df6, df7, df8, df9], ignore_index=True)
df.to_csv("comparison.csv")

df1["timestamp"] = df1["time_in_s"] * 3
df1 = df1.round({"timestamp": 0})
df1 = df1.astype({'timestamp': 'int', "try": "int"})
df1.drop("time_in_s", inplace=True, axis=1)
df2["timestamp"] = df2["time_in_s"] * 3
df2 = df2.round({"timestamp": 0})
df2 = df2.astype({'timestamp': 'int', "try": "int"})
df2.drop("time_in_s", inplace=True, axis=1)
df4["timestamp"] = df4["time_in_s"] * 3
df4 = df4.round({"timestamp": 0})
df4 = df4.astype({'timestamp': 'int', "try": "int"})
df4.drop("time_in_s", inplace=True, axis=1)
df5["timestamp"] = df5["time_in_s"] * 3
df5 = df5.round({"timestamp": 0})
df5 = df5.astype({'timestamp': 'int', "try": "int"})
df5.drop("time_in_s", inplace=True, axis=1)
df6["timestamp"] = df6["time_in_s"] * 3
df6 = df6.round({"timestamp": 0})
df6 = df6.astype({'timestamp': 'int', "try": "int"})
df6.drop("time_in_s", inplace=True, axis=1)
df7["timestamp"] = df7["time_in_s"] * 3
df7 = df7.round({"timestamp": 0})
df7 = df7.astype({'timestamp': 'int', "try": "int"})
df7.drop("time_in_s", inplace=True, axis=1)
df8["timestamp"] = df8["time_in_s"] * 3
df8 = df8.round({"timestamp": 0})
df8 = df8.astype({'timestamp': 'int', "try": "int"})
df8.drop("time_in_s", inplace=True, axis=1)
df9["timestamp"] = df9["time_in_s"] * 3
df9 = df9.round({"timestamp": 0})
df9 = df9.astype({'timestamp': 'int', "try": "int"})
df9.drop("time_in_s", inplace=True, axis=1)
max_length = max(df2.groupby('try')['timestamp'].max().max(), df1.groupby('try')['timestamp'].max().max())


# Define a function to pad each group
def pad_time_series(group):
    #max_length = group['timestamp'].max()  # Find the maximum timestamp
    last_Power = group.iloc[-1]  # Get the last row
    index_range = range(0, max_length + 1)  # Create the range of indices
    padded_series = pd.DataFrame({'timestamp': index_range})  # Create a DataFrame with the desired range
    padded_series = pd.merge(padded_series, group, on='timestamp',
                             how='left')  # Merge with the original group to fill Powers
    padded_series['Power'] = padded_series['Power'].fillna(method='ffill')  # Fill NaNs with the last non-null Power
    padded_series['try'] = padded_series['try'].fillna(method='ffill')  # Fill NaNs with the last non-null try
    padded_series['reset_number'] = padded_series['reset_number'].fillna(
        method='ffill')  # Fill NaNs with the last non-null reset_number
    return padded_series


# Apply the function to each group
padded_df1 = df1.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df2 = df2.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df4 = df4.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df5 = df5.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df6 = df6.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df7 = df7.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df8 = df8.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df9 = df9.groupby('try').apply(pad_time_series).reset_index(drop=True)
padded_df2["agent"] = "0.9_TQC_pretrained"
padded_df1["agent"] = "0.9_TQC"
padded_df4["agent"] = "0.87_TQC"
padded_df5["agent"] = "0.85_TQC"
padded_df6["agent"] = "0.85_CrossQ"
padded_df7["agent"] = "0.9_CrossQ"
padded_df8["agent"] = "0.85_SAC"
padded_df9["agent"] = "0.9_SAC"
#padded_df10 = pd.read_csv(f"human/padded_human_testing.csv", index_col=0)
#padded_df10["agent"] = "human"

padded_df = pd.concat([padded_df1, padded_df2, padded_df4, padded_df5, padded_df6, padded_df7, padded_df8,
                       padded_df9], ignore_index=True)
padded_df["rounded_time_in_s"] = padded_df["timestamp"] / 3
padded_df = padded_df.round({"rounded_time_in_s": 1})
print(padded_df)
padded_df.to_csv("padded_comparison.csv")
padded_df["std1"] = 0.01 * padded_df["Power"]
padded_df["mean_squared_plus_std1_squared"] = (padded_df["Power"].apply(lambda x: x ** 2)
                                               + padded_df["std1"].apply(lambda x: x ** 2))


def std_combined(group):
    mean_means = group["Power"].mean()
    std = np.sqrt(group["mean_squared_plus_std1_squared"].sum() / (group["Power"].size)
                  - mean_means ** 2)
    y_upper = mean_means + 2 * std
    y_lower = mean_means - 2 * std
    if y_upper > 1:
        y_upper = 1
    if y_lower < 0:
        y_lower = 0
    return pd.Series({"mean": mean_means, "std": std, "y_upper": y_upper, "y_lower": y_lower})


stats_df = padded_df.groupby(['rounded_time_in_s', 'agent']).apply(std_combined).reset_index()

print(stats_df)
stats_df.to_csv("stats_comparison.csv")

sns.lineplot(x='rounded_time_in_s', y='mean', data=stats_df, hue="agent", legend=True)
categories = stats_df['agent'].unique()
for category in categories:
    subset = stats_df[stats_df['agent'] == category]
    plt.fill_between(subset['rounded_time_in_s'], subset['y_lower'], subset['y_upper'], alpha=0.2)

plt.show()
