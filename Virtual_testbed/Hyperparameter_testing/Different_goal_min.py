import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns


sns.set(font_scale=1, rc={'text.usetex' : True})
filename = "TQC_reward_2024_04_22_betas_5_5_5_5_1_alphas_0.5_0.9_0.5_prefactors_10_10_100_P_goal_fail_min"
df = pd.read_csv(filename+".csv", index_col=0)
df["steps_in_10000"] = df["number_timesteps"]/10000
df = df.astype({"steps_in_10000": int})
savefile = "TQC_reward_2024_04_22_betas_5_5_5_5_1_alphas_0.5_0.9_0.5_prefactors_10_10_100_P_goal_fail_min"

print(df)

hyperparameter_plot_in_goal = sns.catplot(data=df, col="reset_power_goal", row="reset_power_fail",
                      margin_titles=True, hue="min_power_after_reset", x="steps_in_10000", y="percentage_in_goal",
                                          kind="point")
hyperparameter_plot_in_goal.set_axis_labels(x_var="training step in $10000$", y_var="$p[goal]$")
hyperparameter_plot_in_goal.figure.savefig(savefile+"_percentage_in_goal.png")

hyperparameter_plot_failed = sns.catplot(data=df, col="reset_power_goal", row="reset_power_fail",
                      margin_titles=True, hue="min_power_after_reset", x="steps_in_10000", y="percentage_failed", kind="point")
hyperparameter_plot_failed.set_axis_labels(x_var="training step in $10000$", y_var="$p[fail]$")
hyperparameter_plot_failed.figure.savefig(savefile+"_percentage_failed.png")

hyperparameter_plot_power = sns.catplot(data=df, col="reset_power_goal", row="reset_power_fail",
                      margin_titles=True, hue="min_power_after_reset", x="steps_in_10000", y="P_max_episode_steps_ave", kind="point")
hyperparameter_plot_power.set_axis_labels(x_var="training step in $10000$", y_var="$P_T$")
hyperparameter_plot_power.figure.savefig(savefile+"_power_end_ave.png")
