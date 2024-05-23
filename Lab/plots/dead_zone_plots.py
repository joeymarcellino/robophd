import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import seaborn as sns

sns.set(font_scale=2.5, rc={'text.usetex': True, "font.family": "serif", "font.sans-serif": "Times"})

color = sns.color_palette("Set1")[2]
df = pd.read_csv("backlash_study.csv", index_col=0)

f, ax = plt.subplots(figsize=(7, 7))
ax.set_yscale("log")

ax = sns.boxplot(data=df, x="actuator", y="Difference start-end", whis=[0, 100], width=0.6, color=color, ax=ax)
ax = sns.stripplot(data=df, x="actuator", y="Difference start-end", size=3, color=".3", ax=ax)
ax.set_xlabel("actuator")
ax.set_xticklabels(["1x", "2x", "1y", "2y"])
ax.set_ylabel("dead-zone size [actuator steps]")
f.figure.savefig("backlash_study25.png", bbox_inches='tight')
plt.show()