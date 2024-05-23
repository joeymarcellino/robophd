import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df0 = pd.read_csv("return_training_pretrained_virtual_stats.csv", index_col=0)
df1 = pd.read_csv("return_replay_buffer_stats_without_replay.csv", index_col=0)
df = pd.concat([df0, df1], ignore_index=True)
print(df1)

df_stats = pd.read_csv("./testing/pretraining_stats.csv", index_col=0)
print(df_stats)
palette = sns.color_palette("colorblind")
palette = dict({"pretrain on lower goals": palette[0], "pretrained on virtual testbed": palette[1],
                'only trained on virtual': palette[2]})
f, ax = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw=dict(width_ratios=[1,1]))

ax[0].set_xlim(0, 219000)


ax[0] = sns.lineplot(data=df, x="Step", y="mean", hue="agent", legend="full",
                          palette=palette, ax=ax[0])
categories = df['agent'].unique()
for category in categories:
    subset = df[df['agent'] == category]
    ax[0].fill_between(subset['Step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette[category])

ax[0].set_xlabel("timesteps")
ax[0].set_ylabel("return (smoothed)")
title0 = ax[0].set_title(r'\textbf{(a)}')
title0.set_position(np.array([-0.15, 0.99]))
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles=handles, labels=labels)
ax[0].plot(204000, 121, marker='*', markersize=10, color=palette["pretrain on lower goals"])
ax[0].plot(218000, 115, marker='*', markersize=10, color=palette["pretrained on virtual testbed"])
ax[0].axvline(x=38000, color="black")
ax[0].axvline(x=63000, color="black")
ax[0].axvline(x=98000, color="black")
#ax[0].legend(title="$P_{goal}$", bbox_to_anchor=(orig_pos.x0+0.58, orig_pos.y0+0.2))



sns.boxplot(x='agent', y='max_time', data=df_stats, whis=[0, 100], ax=ax[1], palette=palette)
sns.stripplot(data=df_stats, x="agent", y="max_time", size=2, color=".4", ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_xticklabels(["only trained\n on virtual testbed", "pretrained on\n lower goals", "pretrained on\n virtual testbed"])
ax[1].set_ylabel("time to reach $P=0.9$ [seconds]")
title5 = ax[1].set_title(r'\textbf{(b)}', fontsize=13)
title5.set_position(np.array([-0.1, 0.99]))

f.figure.savefig("return_appendix_pre_virtual.pdf", format="pdf", bbox_inches='tight')
plt.show()