import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns


sns.set(font_scale=1, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})
filename = "prefactor_tests"
df = pd.read_csv(filename+".csv", index_col=0)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})

print(df)
hyperparameter_plot_in_goal = sns.catplot(data=df, col="prefactor_fail", hue="prefactor_goal",
                      margin_titles=True,  x="steps_in_10000", y="percentage_in_goal",
                                          kind="point", palette=sns.color_palette("colorblind"))
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10^4$", y_var="$P[goal]$")
for ax, title in zip(hyperparameter_plot_in_goal.axes.flat, ['$A_{f}=10$', '$A_{f}=100$', '$A_{f}=1000$']):
    ax.set_title(title)
hyperparameter_plot_in_goal._legend.set_title("$A_g$")
hyperparameter_plot_in_goal.fig.set_figwidth(12)
hyperparameter_plot_in_goal.fig.set_figheight(4)
hyperparameter_plot_in_goal.fig.set_size_inches(12, 4)
hyperparameter_plot_in_goal.figure.savefig(filename+"_goal.pdf", format="pdf", bbox_inches="tight")
plt.show()


hyperparameter_plot_failed = sns.catplot(data=df, col="prefactor_fail", hue="prefactor_goal",
                      margin_titles=True,  x="steps_in_10000", y="percentage_failed",
                                          kind="point", palette=sns.color_palette("colorblind"))
hyperparameter_plot_failed.set_axis_labels(x_var="training step in $10^4$", y_var="$P[fail]$")
for ax, title in zip(hyperparameter_plot_failed.axes.flat, ['$A_{f}=10$', '$A_{f}=100$', '$A_{f}=1000$']):
    ax.set_title(title)
hyperparameter_plot_failed._legend.set_title("$A_g$")
hyperparameter_plot_failed.fig.set_figwidth(12)
hyperparameter_plot_failed.fig.set_figheight(4)
hyperparameter_plot_failed.fig.set_size_inches(12, 4)
hyperparameter_plot_failed.figure.savefig(filename+"_fail.pdf", format="pdf", bbox_inches="tight")
plt.show()

"""

hyperparameter_plot_in_goal = sns.catplot(data=df, hue="prefactor_fail", col="prefactor_goal",
                      margin_titles=True,  x="steps_in_10000", y="percentage_in_goal",
                                          kind="point")
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10000$", y_var="$p[goal]$")
hyperparameter_plot_in_goal.figure.savefig(filename+"_percentage_in_goal.png")

hyperparameter_plot_failed = sns.catplot(data=df, hue="prefactor_fail", col="prefactor_goal",
                      margin_titles=True,  x="steps_in_10000", y="percentage_failed", kind="point")
hyperparameter_plot_failed.set_axis_labels(x_var="training step in $10000$", y_var="$p[fail]$")
hyperparameter_plot_failed.figure.savefig(filename+"_percentage_failed.png")

hyperparameter_plot_power = sns.catplot(data=df, hue="prefactor_fail", col="prefactor_goal",
                      margin_titles=True,  x="steps_in_10000", y="P_max_episode_steps_ave", kind="point")
hyperparameter_plot_power.set_axis_labels(x_var="training step in $10000$", y_var="$P_T$")
hyperparameter_plot_power.figure.savefig(filename+"_power_end_ave.png")
"""
"""
power_plot = sns.relplot(data=df_long, x="step", y="power_", kind="line")
power_plot.set_axis_labels(x_var="$t$", y_var="$P$")
power_plot.figure.savefig(filename+".png")
plt.show()
"""
