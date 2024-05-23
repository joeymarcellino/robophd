import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df_prob_goal_lin = pd.read_csv("P_goal_linear_tests.csv", index_col=0)
df_prob_goal_lin = df_prob_goal_lin.groupby('timestamp').tail(1).reset_index(drop=True)
df_prob_goal_lin["goal_function"] = "linear"
df_prob_goal_step = pd.read_csv("P_goal_step_tests.csv", index_col=0)
df_prob_goal_step = df_prob_goal_step.groupby('timestamp').tail(1).reset_index(drop=True)
df_prob_goal_step["goal_function"] = "step"
df_prob_goal = pd.concat([df_prob_goal_lin, df_prob_goal_step], ignore_index=True)
df_return = pd.read_csv("P_goal_stats.csv", index_col=0)
palette = sns.color_palette("colorblind") #[0:3]+sns.color_palette("colorblind")[-4:-1]
palette_prob_goal = dict({"linear": palette[0], "step": palette[1]})
palette = sns.color_palette('flare', n_colors=8)
palette_return = dict({0.8: palette[0], 0.85: palette[1], 0.86: palette[2], 0.87: palette[3],
                       0.88: palette[4], 0.89: palette[5], 0.9: palette[6], 0.91: palette[7]})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.25), gridspec_kw=dict(width_ratios=[1, 1]))


axes[0] = sns.lineplot(x='rounded_step', y='mean', data=df_return, hue="P_goal", legend="full",
                          palette=palette_return, ax=axes[0])
categories = df_return['P_goal'].unique()
for category in categories:
    subset = df_return[df_return['P_goal'] == category]
    axes[0].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette_return[category])
axes[0].set_ylabel('return (smoothed)')
axes[0].set_xlabel('training steps (rounded to 1000)')
new_labels0 = ['$0.8$', '$0.85$', '$0.86$', '$0.87$', '$0.88$', '$0.89$', '$0.9$', '$0.91$']
handles0, labels0 = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles0, labels=new_labels0, ncol=2, title="$P_{goal}$")
title0 = axes[0].set_title(r'\textbf{(a)}', fontsize=13)
title0.set_position(np.array([-0.125, 0.99]))


for (goal_fct_name, goal_fct) in df_prob_goal.groupby("goal_function"):
    means = goal_fct.groupby('P_goal_start')['percentage_in_goal'].mean()
    errs = goal_fct.groupby('P_goal_start')['percentage_in_goal'].std()
    axes[1].errorbar(means.index, means, marker="o", yerr=errs, lw=2, label=goal_fct_name)
axes[1].set_xlabel('$P_{goal,start}$')
axes[1].set_ylabel('$P[goal]$ for $t=T$ after $10^5$ training steps')
axes[1].legend(title="goal function", loc="upper left")
title1 = axes[1].set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.1, 0.99]))


plt.tight_layout()
fig.figure.savefig("P_goal_plots.pdf", format="pdf", bbox_inches="tight")
plt.show()