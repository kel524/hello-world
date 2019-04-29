#-*- coding: utf-8 -*-

"""
    Author: Changwoo Kim (KETI) / cwkim@keti.re.kr / +82-10-9536-0610
    data_preprocessing.py : parsing data exported from machbase DB
"""

# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Data manipulation, visualization and useful functions
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 50
pd.options.display.max_columns = 40
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling algorithms
# General(Statistics/Econometrics)
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error,  r2_score, mean_absolute_error
# for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Customized
from multiprocessing import Process, Queue
import sys, os, time, logging, pprint
from logging import handlers
from datetime import datetime
import pdb

#########################################################################
# Timestamp to datetime for Machbase DB
def timestamp2datetime(df):
    # Translate TIMESTAMP to DATETIME
    for i, row in enumerate(df['TIMESTAMP']):
        df['TIMESTAMP'][i] = pd.to_datetime(row).strftime("%Y-%m-%d %H:%M:00")
    return df

# Tag with own file data name
def tagging(df, tag):
    columns_with_tag = list()
    for column in df.columns:
        if (column == 'TIMESTAMP'):# or (column == 'DATETIME'):
            columns_with_tag.append(column)
            continue 
        columns_with_tag.append(tag+'_'+column)   
    df.columns = columns_with_tag
    return df

#########################################################################
# Static dir path
resource_dir = os.getcwd() + '/ALL_STM_20190320-20190403/'

# Make result directory
if not(os.path.isdir(resource_dir+'result')):
    os.makedirs(os.path.join(resource_dir+'result'))

for csv in os.listdir(resource_dir):
    # Check if the file extenstion is csv
    if os.path.splitext(csv)[-1] == '.csv':
        # Read CSV file contents
        df = pd.read_csv(resource_dir+csv, engine='python')
        df = df.sort_values(by=['TIMESTAMP'], ascending=True)
        df = timestamp2datetime(df)
        # Propagate non-null values forward and backward
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        # averaging rows that include same minutes in TIMESTAMP column
        df = df.groupby(['TIMESTAMP'], as_index=False).mean()

        # Check if the file from ONP_DISP
        if 'ONP' in os.path.splitext(csv)[0]:
            df_onp = tagging(df, 'ONP')
            df.to_csv(path_or_buf=resource_dir+'result/'+'TAG_'+csv, sep=',', na_rep='NaN', index=False)
            print("Saved "+resource_dir+'result/'+'TAG_'+csv)
        # Check if the file from OMG_DISP
        elif 'OMG' in os.path.splitext(csv)[0]:
            df_omg = tagging(df, 'OMG')
            df.to_csv(path_or_buf=resource_dir+'result/'+'TAG_'+csv, sep=',', na_rep='NaN', index=False)
            print("Saved "+resource_dir+'result/'+'TAG_'+csv)
        # Check if the file from OCC_DISP
        elif 'OCC' in os.path.splitext(csv)[0]:
            df_occ = tagging(df, 'OCC')
            df.to_csv(path_or_buf=resource_dir+'result/'+'TAG_'+csv, sep=',', na_rep='NaN', index=False)
            print("Saved "+resource_dir+'result/'+'TAG_'+csv)
        else:
            break
            
print("Done")
