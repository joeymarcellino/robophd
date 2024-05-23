import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np



sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df85 = pd.read_csv("algorithm85_stats.csv", index_col=0)
df90 = pd.read_csv("algorithm90_stats.csv", index_col=0)

palette = sns.color_palette("colorblind")[0:3]+sns.color_palette("colorblind")[-4:-1]
print(palette)
palette = dict({"SAC": palette[0], "PPO": palette[1], "TQC": palette[2],
                "DDPG": palette[3], "TD3": palette[5], "A2C": palette[4]})

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=[2, 2, 0.5]), constrained_layout=True)


axes[0] = sns.lineplot(x='rounded_step', y='mean', data=df85, hue="algorithm", legend=False,
                          palette=palette, ax=axes[0])
categories = df85['algorithm'].unique()
for category in categories:
    subset = df85[df85['algorithm'] == category]
    axes[0].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette[category])
axes[0].set_ylabel('return (smoothed)')
axes[0].set_xlabel('training steps (rounded to 500)')
title0 = axes[0].set_title(r'\textbf{(a)}', fontsize=13)
title0.set_position(np.array([-0.125, 0.99]))


axes[1] = sns.lineplot(x='rounded_step', y='mean', data=df90, hue="algorithm", legend=False,
                          palette=palette, ax=axes[1])
categories = df90['algorithm'].unique()
for category in categories:
    subset = df90[df90['algorithm'] == category]
    axes[1].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette[category])
axes[1].set_ylabel('return (smoothed)')
axes[1].set_xlabel('training steps (rounded to 1000)')
title1 = axes[1].set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.125, 0.99]))


orig_pos = axes[2].get_position(original=True)
labels = ['SAC', 'TQC', 'TD3', 'DDPG', 'PPO', 'A2C']
handles = [Line2D([], [], color=palette["SAC"]), Line2D([], [], color=palette["TQC"]),
           Line2D([], [], color=palette["TD3"]), Line2D([], [], color=palette["DDPG"]),
           Line2D([], [], color=palette["PPO"]), Line2D([], [], color=palette["A2C"])]
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(orig_pos.x0+0.08, orig_pos.y0+0.45))
fig.canvas.draw()
fig.delaxes(axes[2])


fig.figure.savefig("algorithm_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
