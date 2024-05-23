import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
df_act = pd.read_csv("max_action_stats.csv", index_col=0)
df_obs = pd.read_csv("number_obs_stats.csv", index_col=0)
palette = sns.color_palette("colorblind") #[0:3]+sns.color_palette("colorblind")[-4:-1]
print(palette)
liste =[(2000, palette[0])]
for i in range(1, 8):
    liste.append(((i+3)*10**3, palette[i]))
palette_act = dict(liste)
liste =[]
for i in range(6):
    liste.append((i+1, palette[i]))
palette_obs = dict(liste)


fig, axes = plt.subplots(1, 2, figsize=(10, 4.25), gridspec_kw=dict(width_ratios=[1, 1]))



axes[0] = sns.lineplot(x='rounded_step', y='mean', data=df_obs, hue="number_obs", legend="full",
                          palette=palette_obs, ax=axes[0])
categories = df_obs['number_obs'].unique()
for category in categories:
    subset = df_obs[df_obs['number_obs'] == category]
    axes[0].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette_obs[category])
axes[0].set_ylabel('return (smoothed)')
axes[0].set_xlabel('training steps (rounded to 500)')
axes[0].legend(ncol=2, title="history length $n$")
title0 = axes[0].set_title(r'\textbf{(a)}', fontsize=13)
title0.set_position(np.array([-0.125, 0.99]))


axes[1] = sns.lineplot(x='rounded_step', y='mean', data=df_act, hue="max_action", legend="full",
                          palette=palette_act, ax=axes[1])
categories = df_act['max_action'].unique()
for category in categories:
    subset = df_act[df_act['max_action'] == category]
    axes[1].fill_between(subset['rounded_step'], subset['y_lower'], subset['y_upper'], alpha=0.2,
                         color=palette_act[category])
axes[1].set_ylabel('return (smoothed)')
axes[1].set_xlabel('training steps (rounded to 500)')
title1 = axes[1].set_title(r'\textbf{(b)}', fontsize=13)
title1.set_position(np.array([-0.1, 0.99]))
new_labels1 = ['$2$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$', '$10$']
handles1, labels1 = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles1, labels=new_labels1, ncol=2, title="max. action $a_{max}$ [$10^3$]", loc="upper left")

plt.tight_layout()
fig.figure.savefig("obs_action_plots.pdf", format="pdf", bbox_inches="tight")
plt.show()