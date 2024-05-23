import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv(f"goal0.85_fail0.05_start0.2.csv", index_col=0)
df["timestamp"] = df["time_in_s"]*3
df = df.round({"timestamp": 0})
df = df.astype({'timestamp': 'int'})
print(df)
max_length = df.groupby('try')['timestamp'].max().max()
# Define a function to pad each group
def pad_time_series(group):
    #max_length = group['timestamp'].max()  # Find the maximum timestamp
    last_Power = group.iloc[-1]            # Get the last row
    index_range = range(0, max_length+1) # Create the range of indices
    padded_series = pd.DataFrame({'timestamp': index_range}) # Create a DataFrame with the desired range
    padded_series = pd.merge(padded_series, group, on='timestamp', how='left') # Merge with the original group to fill Powers
    padded_series['Power'] = padded_series['Power'].fillna(method='ffill') # Fill NaNs with the last non-null Power
    padded_series['try'] = padded_series['try'].fillna(method='ffill') # Fill NaNs with the last non-null try
    return padded_series

# Apply the function to each group
padded_df = df.groupby('try').apply(pad_time_series).reset_index(drop=True)

padded_df["rounded_time_in_s"] = padded_df["timestamp"]/3
padded_df = padded_df.round({"rounded_time_in_s": 1})
padded_df.drop("time_in_s", inplace=True, axis=1)
padded_df["agent"] = "human"
print(padded_df)
padded_df.to_csv("padded_human_testing.csv")


sns.relplot(data=padded_df, x="rounded_time_in_s", y="Power", kind="line")
plt.show()