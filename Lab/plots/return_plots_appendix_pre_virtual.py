import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df0 = pd.read_csv("return_training_pretrained_virtual_stats.csv", index_col=0)
df2 = pd.read_csv("return_training_pretrained_virtual_noise_stats.csv", index_col=0)
df1 = pd.read_csv("return_replay_buffer_stats_without_replay.csv", index_col=0)
df = pd.concat([df0, df1], ignore_index=True)
df = pd.concat([df, df2], ignore_index=True)
print(df1)

df_stats = pd.read_csv("./testing/pretraining_stats.csv", index_col=0)
print(df_stats)
palette = sns.color_palette("colorblind")
palette = dict({"pretrain on lower goals": palette[0], "pretrained on virtual testbed": palette[1],
                'only trained on virtual': palette[2], "pretrained on virtual testbed with noise": palette[3],
                'only trained on virtual with noise': palette[4]})
f, ax = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw=dict(width_ratios=[2, 1.5, 0.75]),
                     constrained_layout=True)

ax[0].set_xlim(0, 219000)


ax[0] = sns.lineplot(data=df, x="Step", y="mean_norm", hue="agent", legend=False,
                          palette=palette, ax=ax[0])
categories = df['agent'].unique()
for category in categories:
    subset = df[df['agent'] == category]
    ax[0].fill_between(subset['Step'], subset['y_lower_norm'], subset['y_upper_norm'], alpha=0.2,
                         color=palette[category])

ax[0].set_xlim(-4000, 223000)
ax[0].set_xlabel("timesteps")
ax[0].set_ylabel("normalized return (smoothed)")
title0 = ax[0].set_title(r'\textbf{(a)}')
title0.set_position(np.array([-0.1, 0.99]))
ax[0].plot(204000, 0.66, marker='*', markersize=10, color=palette["pretrain on lower goals"])
ax[0].plot(0, 0.61, marker='*', markersize=10, color=palette["only trained on virtual"])
ax[0].plot(218000, 0.64, marker='*', markersize=10, color=palette["pretrained on virtual testbed"])
ax[0].plot(0, 0.66, marker='*', markersize=10, color=palette["only trained on virtual with noise"])
ax[0].plot(136000, 0.72, marker='*', markersize=10, color=palette["pretrained on virtual testbed with noise"])
ax[0].axvline(x=38000, color="black")
ax[0].axvline(x=63000, color="black")
ax[0].axvline(x=98000, color="black")



sns.boxplot(x="agent", y='max_time', data=df_stats, whis=[0, 100], ax=ax[1], palette=palette, hue="agent", legend=False)
sns.stripplot(data=df_stats, x="agent", y="max_time", size=2, color=".4", ax=ax[1])
ax[1].set_xlabel("")
ax[1].set_xticklabels(["", "", "", "", ""])
ax[1].set_ylabel("time to reach $P=0.9$ [seconds]")
title5 = ax[1].set_title(r'\textbf{(b)}', fontsize=13)
title5.set_position(np.array([-0.125, 0.99]))

labels = ['pretrained on\nlower goals', 'pretrained on\nvirtual testbed', 'only trained\non virtual',
          'pretrained on\nvirtual testbed\nwith noise', 'only trained\non virtual\nwith noise']
handles = [Line2D([], [], color=palette["pretrain on lower goals"]),
           Line2D([], [], color=palette["pretrained on virtual testbed"]),
           Line2D([], [], color=palette["only trained on virtual"]),
           Line2D([], [], color=palette["pretrained on virtual testbed with noise"]),
           Line2D([], [], color=palette["only trained on virtual with noise"])]

orig_pos = ax[2].get_position(original=True)
legend = f.legend(handles, labels, loc='center left', bbox_to_anchor=(orig_pos.x0+0.06, orig_pos.y0+0.4))
f.delaxes(ax[2])
f.figure.savefig("return_appendix_pre_virtual.pdf", format="pdf", bbox_inches='tight')
plt.show()