import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
filename1 = "alphas_betas_step1"
df1 = pd.read_csv(filename1+".csv", index_col=0)
filename2 = "alphas_betas_step2"
df2 = pd.read_csv(filename2+".csv", index_col=0)
df = pd.concat([df1, df2], ignore_index=True)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})
savefile = "TQC_reward_2024_04_22_alpha_beta_search_step"

hyperparameter_plot_in_goal = sns.catplot(data=df, col="beta_step", hue="alpha_step",
                                          x="steps_in_10000", y="percentage_in_goal",
                                          kind="point", palette=sns.color_palette("colorblind"))
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10^4$", y_var="$P[goal]$")
for ax, title in zip(hyperparameter_plot_in_goal.axes.flat, [r'$\beta_{s}=1$',
                                                             r'$\beta_{s}=5$',
                                                             r'$\beta_{s}=10$']):
    ax.set_title(title)
hyperparameter_plot_in_goal._legend.set_title(r"$\alpha_s$")
hyperparameter_plot_in_goal.fig.set_figwidth(12)
hyperparameter_plot_in_goal.fig.set_figheight(3)
hyperparameter_plot_in_goal.fig.set_size_inches(12, 3)
hyperparameter_plot_in_goal.figure.savefig(savefile+"_goal.pdf", format="pdf", bbox_inches="tight")
plt.show()


filename = "alphas_betas_goal"
df = pd.read_csv(filename+".csv", index_col=0)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})
savefile = "TQC_reward_2024_04_22_alpha_beta_search_goal"

hyperparameter_plot_in_goal = sns.catplot(data=df, col="beta_goal_1", row="beta_goal_2", hue="alpha_goal",
                                          x="steps_in_10000", y="percentage_in_goal",
                                          kind="point", palette=sns.color_palette("colorblind"))
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10^4$", y_var="$P[goal]$")
for ax, title in zip(hyperparameter_plot_in_goal.axes.flat, [r'$\beta_{g1}=1$, $\beta_{g2}=1$',
                                                             r'$\beta_{g1}=5$, $\beta_{g2}=1$',
                                                             r'$\beta_{g1}=1$, $\beta_{g2}=5$',
                                                             r'$\beta_{g1}=5$, $\beta_{g2}=5$']):
    ax.set_title(title)
hyperparameter_plot_in_goal._legend.set_title(r"$\alpha_g$")
hyperparameter_plot_in_goal.fig.set_figwidth(10)
hyperparameter_plot_in_goal.fig.set_figheight(6)
hyperparameter_plot_in_goal.fig.set_size_inches(10, 6)
hyperparameter_plot_in_goal.figure.savefig(filename+"_goal.pdf", format="pdf", bbox_inches="tight")
plt.show()


filename = "alphas_betas_fail"
df = pd.read_csv(filename+".csv", index_col=0)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})
savefile = "TQC_reward_2024_04_22_alpha_beta_search_goal"

hyperparameter_plot_in_goal = sns.catplot(data=df, col="beta_fail_1", row="beta_fail_2", hue="alpha_fail",
                                          x="steps_in_10000", y="percentage_in_goal",
                                          kind="point", palette=sns.color_palette("colorblind"))
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10^4$", y_var="$P[goal]$")
for ax, title in zip(hyperparameter_plot_in_goal.axes.flat, [r'$\beta_{f1}=1$, $\beta_{f2}=1$',
                                                             r'$\beta_{f1}=5$, $\beta_{f2}=1$',
                                                             r'$\beta_{f1}=1$, $\beta_{f2}=5$',
                                                             r'$\beta_{f1}=5$, $\beta_{f2}=5$']):
    ax.set_title(title)
hyperparameter_plot_in_goal._legend.set_title(r"$\alpha_f$")
hyperparameter_plot_in_goal.fig.set_figwidth(10)
hyperparameter_plot_in_goal.fig.set_figheight(6)
hyperparameter_plot_in_goal.fig.set_size_inches(10, 6)
hyperparameter_plot_in_goal.figure.savefig(filename+"_goal.pdf", format="pdf", bbox_inches="tight")
plt.show()