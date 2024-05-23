import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df0 = pd.read_csv("return_start_from_0_stats.csv", index_col=0)
df1 = pd.read_csv("return_replay_buffer_stats.csv", index_col=0)

palette1 = sns.color_palette("colorblind")
palette1 = dict({"0.85": palette1[0], "0.875_without_replay": palette1[1], "0.875_with_replay": palette1[2],
                "0.89_without_replay": palette1[1], "0.89_with_replay": palette1[2],
                "0.9_without_replay": palette1[1], "0.9_with_replay": palette1[2]})

palette0 = sns.color_palette('flare', n_colors=6)
palette0 = dict({0.85: palette0[0], 0.86: palette0[1], 0.87: palette0[2], 0.88: palette0[3], 0.9: palette0[5]})
f, ax = plt.subplots(1, 4, figsize=(10, 3), gridspec_kw=dict(width_ratios=[3, 1, 3, 0.5]))

ax[0].set_xlim(0, 60000)


ax[0] = sns.lineplot(data=df0, x="Step", y="mean", hue="goal_power", legend="full",
                          palette=palette0, ax=ax[0])
categories = df0['goal_power'].unique()
for category in categories:
    subset = df0[df0['goal_power'] == category]
    ax[0].fill_between(subset['Step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette0[category])

ax[0].set_xlabel("timesteps")
ax[0].set_ylabel("return (smoothed)")
title0 = ax[0].set_title(r'\textbf{(a)}')
title0.set_position(np.array([-0.15, 0.99]))
orig_pos = ax[1].get_position(original=True)
f.delaxes(ax[1])
ax[0].legend(title="$P_{goal}$", bbox_to_anchor=(orig_pos.x0+0.58, orig_pos.y0+0.2))

ax[2].set_xlim(0, 210000)


ax[2] = sns.lineplot(data=df1, x="Step", y="mean", hue="goal_power_replay", legend=False,
                          palette=palette1, ax=ax[2])
categories = df1['goal_power_replay'].unique()
for category in categories:
    subset = df1[df1['goal_power_replay'] == category]
    ax[2].fill_between(subset['Step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette1[category])
ax[2].set_xlabel("timesteps")
ax[2].set_ylabel("return (smoothed)")
title2 = ax[2].set_title(r'\textbf{(b)}')
ax[2].axvline(x=38000, color="black")
ax[2].axvline(x=63000, color="black")
ax[2].axvline(x=98000, color="black")
title2.set_position(np.array([-0.1, 0.99]))
orig_pos = ax[3].get_position(original=True)
labels2 = ['start', 'with replay', 'without replay']
handles2 = [Line2D([], [], color=palette1["0.85"]),
            Line2D([], [], color=palette1["0.89_with_replay"]),
           Line2D([], [], color=palette1["0.89_without_replay"])]
legend = f.legend(handles2, labels2, loc='center left', bbox_to_anchor=(orig_pos.x0-0.03, orig_pos.y0+0.4))
f.canvas.draw()
f.delaxes(ax[3])

f.figure.savefig("return_appendix_replay_goal.pdf", format="pdf", bbox_inches='tight')
plt.show()