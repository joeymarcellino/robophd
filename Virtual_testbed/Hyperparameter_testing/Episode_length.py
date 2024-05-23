import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})


filename = "episode_length_tests"
df_episode_steps = pd.read_csv(filename+".csv", index_col=0)
df_episode_steps["max_actioninsteps"] = 6000
filename = "episode_length_max_action_in_steps_tests"
df_eps_max_act = pd.read_csv(filename+".csv", index_col=0)
df = pd.concat([df_episode_steps, df_eps_max_act], ignore_index=True)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})
savefile = "episode_length_max_action_in_steps_plots"

palette = sns.color_palette("colorblind")
print(len(palette))
palette = dict({5: palette[0], 10: palette[1], 15: palette[2], 20: palette[3], 25: palette[4],
                 30: palette[5], 35: palette[6], 40: palette[7], 45: palette[8], 50: palette[9]})

hyperparameter_plot_in_goal = sns.catplot(data=df, col="max_actioninsteps",
                      margin_titles=True, hue="max_episode_steps", x="steps_in_10000", y="percentage_in_goal",
                                          kind="point", palette="flare", facet_kws={'legend_out': True},
                                          #linestyles='-', markers='o', markersize=0.001
                                          )
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10^4$", y_var="$P[goal]$")
for ax, title in zip(hyperparameter_plot_in_goal.axes.flat, ['$a_{max}=2000$', '$a_{max}=6000$', '$a_{max}=10000$']):
    ax.set_title(title)
hyperparameter_plot_in_goal._legend.set_title("history length $T$")
hyperparameter_plot_in_goal.fig.set_figwidth(12)
hyperparameter_plot_in_goal.fig.set_figheight(4)
hyperparameter_plot_in_goal.fig.set_size_inches(12, 4)
hyperparameter_plot_in_goal.figure.savefig(savefile+".pdf", format="pdf", bbox_inches="tight")
plt.show()


