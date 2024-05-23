import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
matplotlib.use('TkAgg')
sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df_alg = pd.read_csv("algorithm80_stats.csv", index_col=0)
df_obs = pd.read_csv("diff_obs_stats.csv", index_col=0)
palette = sns.color_palette("colorblind")[0:3]+sns.color_palette("colorblind")[-4:-1]
print(palette)
palette_algs = dict({"SAC": palette[0], "PPO": palette[1], "TQC": palette[2],
                "DDPG": palette[3], "TD3": palette[5], "A2C": palette[4]})

palette_obs = dict({"P_ave_0_P_max_0_x_max_0": palette[0], "P_ave_0_P_max_1_x_max_1": palette[1],
                    "P_ave_1_P_max_0_x_max_0": palette[5], "P_ave_1_P_max_1_x_max_1": palette[2]})

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), gridspec_kw=dict(width_ratios=[3, 3, 0.5]), constrained_layout=True)


axes[0] = sns.lineplot(data=df_obs, x="rounded_step", y="mean", hue="obs_kind", legend="full",
                          palette=palette_obs, ax=axes[0])
categories = df_obs['obs_kind'].unique()
for category in categories:
    subset = df_obs[df_obs['obs_kind'] == category]
    axes[0].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette_obs[category])
axes[0].set_ylabel('return (smoothed)')
axes[0].set_xlabel('training steps (rounded to 500)')
new_labels0 = ['without $P_{ave}, P_{max}, x_{max}$', 'incl. $P_{max}, x_{max}$', 'incl. $P_{ave}$',
  'incl. $P_{ave}, P_{max}, x_{max}$']
handles0, labels0 = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles0, labels=new_labels0, title="observation")
title0 = axes[0].set_title(r'\textbf{(a)}', fontsize=13)
title0.set_position(np.array([-0.125, 0.99]))




axes[1] = sns.lineplot(x='rounded_step', y='mean', data=df_alg, hue="algorithm", legend=False,
                          palette=palette_algs, ax=axes[1])
categories = df_alg['algorithm'].unique()
for category in categories:
    subset = df_alg[df_alg['algorithm'] == category]
    axes[1].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette_algs[category])
axes[1].set_ylabel('return (smoothed)')
axes[1].set_xlabel('training steps (rounded to 500)')
title1 = axes[1].set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.1, 0.99]))

orig_pos = axes[2].get_position(original=True)
labels = ['SAC', 'TQC', 'TD3', 'DDPG', 'PPO', 'A2C']
handles = [Line2D([], [], color=palette_algs["SAC"]), Line2D([], [], color=palette_algs["TQC"]),
           Line2D([], [], color=palette_algs["TD3"]), Line2D([], [], color=palette_algs["DDPG"]),
           Line2D([], [], color=palette_algs["PPO"]), Line2D([], [], color=palette_algs["A2C"])]
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(orig_pos.x0+0.1, orig_pos.y0+0.45))
fig.canvas.draw()
fig.delaxes(axes[2])

#plt.tight_layout()
fig.figure.savefig("virtual_testbed_main_paper_plots.pdf", format="pdf", bbox_inches="tight")
plt.show()



