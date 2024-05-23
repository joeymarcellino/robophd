import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np


sns.set(font_scale=1, rc={'text.usetex' : True})
df = pd.read_csv("reset_test_stats.csv", index_col=0)

df = df.groupby('reset_method').filter(lambda x: ((x['reset_method'] == "Interval21000").any() or
                                                                   (x['reset_method'] == "MovePowerUp").any() or
                                                                   (x['reset_method'] == "MovePowerUp1").any()))

df_start = pd.read_csv("reset_tests_probabilities.csv", index_col=0)
df_start = df_start.groupby('reset_method').filter(lambda x: (not (x['reset_method'] == "interval24000").any()))

palette = sns.color_palette("colorblind")[0:3]+sns.color_palette("colorblind")[-4:-1]
print(palette)
palette = dict({"interval21000": palette[0], "move_power_up10_0": palette[1], "move_power_up10_100000": palette[1],
                "move_power_up1": palette[2], "Interval21000": palette[0], "MovePowerUp": palette[1],
                "MovePowerUp1": palette[2]})

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=[2, 2, 0.5]), constrained_layout=True)


axes[0] = sns.boxplot(x='reset_method', y='Power', data=df_start, whis=[0, 100], ax=axes[0], palette=palette)
axes[0] = sns.stripplot(data=df_start, x='reset_method', y='Power', size=2, color=".4", ax=axes[0])
axes[0].set_xlabel("reset method")
axes[0].set_xticklabels(["B", "A,\n start", "A,\n after training", "C"])
axes[0].set_ylabel("$P_0$")
title0 = axes[0].set_title(r'\textbf{(a)}', fontsize=13)
title0.set_position(np.array([-0.125, 0.99]))



axes[1] = sns.lineplot(x='rounded_step', y='mean', data=df, hue="reset_method", legend=False,
                          palette=palette, ax=axes[1])
categories = df['reset_method'].unique()
for category in categories:
    subset = df[df['reset_method'] == category]
    axes[1].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette[category])
axes[1].set_ylabel('return (smoothed)')
axes[1].set_xlabel('training steps (rounded to 500)')
title1 = axes[1].set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.125, 0.99]))


orig_pos = axes[2].get_position(original=True)
labels = ['A', 'B', 'C']
handles = [Line2D([], [], color=palette["move_power_up10_0"]),
           Line2D([], [], color=palette["move_power_up1"]), Line2D([], [], color=palette["interval21000"])]
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(orig_pos.x0+0.08, orig_pos.y0+0.45))
fig.canvas.draw()
fig.delaxes(axes[2])


fig.figure.savefig("reset_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
"""

print(df_start)

sns.boxplot(x='reset_method', y='Power', data=df_start, whis=[0, 100])
sns.stripplot(data=df_start, x='reset_method', y='Power', size=2, color=".4")
plt.show()

"""
"""
sns.displot(data=df_start_acts, x='act1x', y='act1y',  hue='reset_method', kind="kde")
plt.show()
"""
