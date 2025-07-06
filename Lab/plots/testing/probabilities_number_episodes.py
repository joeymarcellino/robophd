import pandas as pd
import numpy as np

df = pd.read_csv("human_vs_agnes_statistics.csv", index_col=0)
df["clipped_reset_number"] = np.clip(df["reset_number"], 0, 3)
# Group by 'agent' and 'clipped_reset_number' and count occurrences
counts = df.groupby(['agent', 'clipped_reset_number']).size().reset_index(name='count')

# Compute probability per agent (common_norm=False => normalize within each agent group)
total_counts = counts.groupby('agent')['count'].transform('sum')
counts['probability'] = counts['count'] / total_counts

# Optional: Pivot to make it easier to view (rows: clipped_reset_number, columns: agent)
prob_table = counts.pivot(index='clipped_reset_number', columns='agent', values='probability').fillna(0)
prob_table.rename(index={3: '3+'}, inplace=True)
print(prob_table)
prob_table.to_csv("probabilities_number_episodes.csv")