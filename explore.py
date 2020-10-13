import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os


def plot_variable_pairs(df):
    g = sns.PairGrid(df) 
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.regplot)

def months_to_years(tenure_months, df):
    df['tenure_years'] = round(tenure_months/12, 0)
    return df

# def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
#     plt.rc('font', size=13)
#     plt.rc('figure', figsize=(13, 7))
#     plt.subplot(311)
#     sns.boxplot(data=df, y=continuous_var, x=categorical_var)
#     plt.subplot(312)
#     sns.violinplot(data=df, y=continuous_var, x=categorical_var)
#     plt.subplot(313)
#     sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
#     plt.tight_layout()
#     plt.show()

def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13, 7))
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()   
    sns.violinplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()