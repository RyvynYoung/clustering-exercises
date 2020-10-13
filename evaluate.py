import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pydataset import data

# Linear Model
from statsmodels.formula.api import ols

from sklearn.metrics import mean_squared_error
from math import sqrt



# create model object
def make_model_object(target, feature, df):
    model_object = ols(f'{target} ~ {feature}', df).fit()
    return model_object

# create evaluation dataframe
def make_evaldf(feature, target, model_object, df):
    evaldf = pd.DataFrame()
    evaldf['feature'] = df[feature]
    evaldf['target'] = df[target]
    evaldf['baseline'] = df[target].mean()
    evaldf['baseline_residual'] = evaldf.baseline - evaldf.target
    evaldf["yhat"] = model_object.predict()
    evaldf["model_residual"] = evaldf.yhat - evaldf.target
    return evaldf

# create dataframe metrics from evaluation datafram
def metrics(df, model_object):
    baseline_sse = (df.baseline_residual**2).sum()
    model_sse = (df.model_residual**2).sum()

    if model_sse < baseline_sse:
        print("Our model beats the baseline")
        print("It makes sense to evaluate this model more deeply.")
    else:
        print("Our baseline is better than the model.")

    print("Baseline SSE", baseline_sse)
    print("Model SSE", model_sse)
    
    mse = mean_squared_error(df.target, df.yhat)
    rmse = sqrt(mse)

    print("MSE is", mse, " which is the average squared error")
    print("RMSE is", rmse, " which is the square root of the MSE")
    
    r2 = model_object.rsquared
    print('R-squared = ', round(r2,3))

    f_pval = model_object.f_pvalue
    print("p-value for model significance = ", f_pval)
    
    alpha = .05
    print(f'f_pval is less than {alpha} = {f_pval<alpha}')
    
# make a plot to visualize residuals
def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()