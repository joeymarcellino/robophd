import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model

df_xm1 = pd.read_csv("moveRange_actxm1.csv", usecols=["Position", "Power"])
df_ym1 = pd.read_csv("moveRange_actym1.csv", usecols=["Position", "Power"])
df_xm2 = pd.read_csv("moveRange_actxm2.csv", usecols=["Position", "Power"])
df_ym2 = pd.read_csv("moveRange_actym2.csv", usecols=["Position", "Power"])

max_act = [df_xm1.max()["Position"], df_ym1.max()["Position"], df_xm2.max()["Position"], df_ym2.max()["Position"]]
min_act = [df_xm1.min()["Position"], df_ym1.min()["Position"], df_xm2.min()["Position"], df_ym2.min()["Position"]]
standardized_df_xm1 = (df_xm1-df_xm1.min())/(df_xm1.max()-df_xm1.min())
standardized_df_ym1 = (df_ym1-df_ym1.min())/(df_ym1.max()-df_ym1.min())
standardized_df_xm2 = (df_xm2-df_xm2.min())/(df_xm2.max()-df_xm2.min())
standardized_df_ym2 = (df_ym2-df_ym2.min())/(df_ym2.max()-df_ym2.min())

def gaussian(x, center, sigma):
    return np.exp(-(x-center)**2/(2*sigma**2))

model_x1 = Model(gaussian)
params_x1 = model_x1.make_params(center=0.5, sigma=0.1)
result_x1 = model_x1.fit(standardized_df_xm1['Power'], params_x1, x=standardized_df_xm1['Position'])
result_x1.plot_fit()
print(result_x1.fit_report())
plt.show()

model_y1 = Model(gaussian)
params_y1 = model_y1.make_params(center=0.5, sigma=0.1)
result_y1 = model_y1.fit(standardized_df_ym1['Power'], params_y1, x=standardized_df_ym1['Position'])
result_y1.plot_fit()
print(result_y1.fit_report())
plt.show()

model_x2 = Model(gaussian)
params_x2 = model_x2.make_params(center=0.5, sigma=0.1)
result_x2 = model_x2.fit(standardized_df_xm2['Power'], params_x2, x=standardized_df_xm2['Position'])
result_x2.plot_fit()
print(result_x2.fit_report())
plt.show()

model_y2 = Model(gaussian)
params_y2 = model_y2.make_params(center=0.5, sigma=0.1)
result_y2 = model_y2.fit(standardized_df_ym2['Power'], params_y2, x=standardized_df_ym2['Position'])
result_y2.plot_fit()
print(result_y2.fit_report())
plt.show()



center_x1 = result_x1.best_values['center']
sigma_x1 = result_x1.best_values['sigma']

center_y1 = result_y1.best_values['center']
sigma_y1 = result_y1.best_values['sigma']

center_x2 = result_x2.best_values['center']
sigma_x2 = result_x2.best_values['sigma']

center_y2 = result_y2.best_values['center']
sigma_y2 = result_y2.best_values['sigma']

param_df = pandas.DataFrame(data={'parameter': ['center_x1', 'sigma_x1', 'center_y1', 'sigma_y1', 'center_x2',
                                                'sigma_x2', 'center_y2', 'sigma_y2'],
                                  'value': [center_x1, sigma_x1, center_y1, sigma_y1, center_x2, sigma_x2, center_y2, sigma_y2]})

param_df.to_csv('2nd_param_fit.csv')
param_df_2 = pandas.DataFrame(data={'parameter': ['max_xm1', 'max_ym1', 'max_xm2', 'max_ym2', 'min_xm1', 'min_ym1',
                                                  'min_xm2', 'min_ym2'],
                                  'value': [df_xm1.max()["Position"], df_ym1.max()["Position"], df_xm2.max()["Position"],
                                            df_ym2.max()["Position"],df_xm1.min()["Position"], df_ym1.min()["Position"],
                                            df_xm2.min()["Position"], df_ym2.min()["Position"]]})
param_df_2.to_csv('2nd_min_max_positions.csv')

"""


def fitted_Gaussian_4D(x1, y1, x2, y2):
    return (np.exp(-(x1-center_x1)**2/(2*sigma_x1**2))
     *np.exp(-(y1-center_y1)**2/(2*sigma_y1**2))
     *np.exp(-(x2-center_x2)**2/(2*sigma_x2**2))
     * np.exp(-(y2 - center_y2) ** 2 / (2 * sigma_y2 ** 2))
     )


df['pd2_pred_standardized'] = np.vectorize(fitted_Gaussian_4D)(df['actxm1_pos_standardized'], df['actym1_pos_standardized'], df['actxm2_pos_standardized'], df['actym2_pos_standardized'])
print(df)
df.to_csv('pred_seperated_fit.csv')

param_df = pandas.DataFrame(data={'parameter': ['center_x1', 'sigma_x1', 'center_y1', 'sigma_y1', 'center_x2', 'sigma_x2', 'center_y2', 'sigma_y2'],
                                  'value': [center_x1, sigma_x1, center_y1, sigma_y1, center_x2, sigma_x2, center_y2, sigma_y2]})
param_df.to_csv('param_fit.csv')
# plot data and fit
df.plot.scatter(x='actxm1_pos_standardized', y='pd2_target_standardized', label='Gridscan data')
plt.scatter(x=df['actxm1_pos_standardized'], y=df['pd2_pred_standardized'], c='red', label='prediction')
plt.ylabel('Power in fiber [arbitrary]')
plt.xlabel('Actuator steps')
plt.legend(loc='upper right')
# plt.savefig('Fit_1x_2.png', dpi=300)
plt.show()
"""