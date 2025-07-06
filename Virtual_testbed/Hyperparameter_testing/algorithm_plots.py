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
df80 = pd.read_csv("algorithm80_stats.csv", index_col=0)
df_act = pd.read_csv("max_action_stats.csv", index_col=0)

palette = sns.color_p"
liste = [(2000, palette[0])]
for i in range(1, 8):
    liste.append(((i+3)*10**3, palette[i]))
palette_act = dict(liste)
palette = sns.color_palette("colorblind")[0:4]+sns.color_palette("colorblind")[-4:-1]
palette = dict({"SAC": palette[0], "PPO": palette[1], "TQC": palette[2],
                "DDPG": palette[3], "TD3": palette[5], "A2C": palette[4], "CrossQ": palette[6]})
fig = plt.figure(figsize=(10, 7), constrained_layout=True)
gs = fig.add_gridspec(4, 9)


ax1 = fig.add_subplot(gs[0:2, 0:5])
sns.lineplot(x='rounded_step', y='mean_norm', data=df_act, hue="max_action", legend="full",
                          palette=palette_act, ax=ax1)
categories = df_act['max_action'].unique()
for category in categories:
    subset = df_act[df_act['max_action'] == category]
    ax1.fill_between(subset['rounded_step'], subset['y_lower_norm'], subset['y_upper_norm'], alpha=0.2,
                         color=palette_act[category])
ax1.set_ylabel('return (smoothed)')
ax1.set_xlabel('training steps (rounded to 500)')
ax1.set_xlim(0, 100000)
title1 = ax1.set_title(r'\textbf{(a)}', fontsize=13)
title1.set_position(np.array([-0.125, 0.99]))
new_labels1 = ['$2$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$', '$10$']
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles=handles1, labels=new_labels1, ncol=2, title="max. action $a_{max}$ [$10^3$]", loc="upper left")


ax2 = fig.add_subplot(gs[0:2, 5:9])
sns.lineplot(x='rounded_step', y='mean_norm', data=df80, hue="algorithm", legend=False,
                          palette=palette, ax=ax2)
categories = df80['algorithm'].unique()
for category in categories:
    subset = df80[df80['algorithm'] == category]
    ax2.fill_between(subset['rounded_step'], subset['y_lower_norm'], subset['y_upper_norm'], alpha=0.2,
                         color=palette[category])
ax2.set_ylabel('return (smoothed)')
ax2.set_xlabel('training steps (rounded to 500)')
ax2.set_xlim(0, 100000)
title1 = ax2.set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.15, 0.99]))

ax3 = fig.add_subplot(gs[2:4, 0:4])
ax3.set_xlim(0, 100000)
sns.lineplot(x='rounded_step', y='mean_norm', data=df85, hue="algorithm", legend=False,
                          palette=palette, ax=ax3)
categories = df85['algorithm'].unique()
for category in categories:
    subset = df85[df85['algorithm'] == category]
    ax3.fill_between(subset['rounded_step'], subset['y_lower_norm'], subset['y_upper_norm'], alpha=0.2,
                         color=palette[category])
ax3.set_ylabel('return (smoothed)')
ax3.set_xlabel('training steps (rounded to 500)')
title0 = ax3.set_title(r'\textbf{(c)}', fontsize=13)
title0.set_position(np.array([-0.15, 0.99]))

ax4 = fig.add_subplot(gs[2:4, 4:8])
# df90 = df90.groupby('algorithm').filter(lambda x: (not (x['algorithm'] == "CrossQ").any())) # use for plotting without CrossQ
sns.lineplot(x='rounded_step', y='mean_norm', data=df90, hue="algorithm", legend=False,
                          palette=palette, ax=ax4)
categories = df90['algorithm'].unique()
for category in categories:
    subset = df90[df90['algorithm'] == category]
    ax4.fill_between(subset['rounded_step'], subset['y_lower_norm'], subset['y_upper_norm'], alpha=0.2,
                         color=palette[category])
ax4.set_ylabel('return (smoothed)')
ax4.set_xlim(0, 500000)
ax4.set_xlabel('training steps (rounded to 500)')
title1 = ax4.set_title(r'\textbf{(d)}', fontsize=13)
title1.set_position(np.array([-0.125, 0.99]))


ax5 = fig.add_subplot(gs[2:4, 8:9])

orig_pos = ax5.get_position(original=True)
labels = ['SAC', 'TQC', 'TD3', 'DDPG', 'PPO', 'A2C', "CrossQ"]
handles = [Line2D([], [], color=palette["SAC"]), Line2D([], [], color=palette["TQC"]),
           Line2D([], [], color=palette["TD3"]), Line2D([], [], color=palette["DDPG"]),
           Line2D([], [], color=palette["PPO"]), Line2D([], [], color=palette["A2C"]),
           Line2D([], [], color=palette["CrossQ"])]
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(orig_pos.x0+0.06, orig_pos.y0+0.15))
fig.canvas.draw()
fig.delaxes(ax5)


fig.figure.savefig("algorithm_action_plots.pdf", format="pdf", bbox_inches="tight")

plt.show()
