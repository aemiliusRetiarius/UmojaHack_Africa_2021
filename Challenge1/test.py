# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

pd.set_option('max_colwidth', 500)

import warnings
warnings.filterwarnings('ignore')
import random

random.seed(42)
np.random.seed(42)

train = pd.read_csv('./Challenge1/Train.csv')
test = pd.read_csv('./Challenge1/Test.csv')
samplesubmission = pd.read_csv('./Challenge1/SampleSubmission.csv')
variable_definations = pd.read_csv('./Challenge1/VariableDefinitions.csv')

ntrain = train.shape[0] # to be used to split train and test set from the combined dataframe

all_data = pd.concat((train, test)).reset_index(drop=True)

# Category columns
cat_cols = ['country',	'region', 'owns_mobile'] + [x for x in all_data.columns if x.startswith('Q')]
num_cols = ['age', 'population']

# Change columns to their respective datatypes
all_data[cat_cols] = all_data[cat_cols].astype('category')

for col in all_data.columns:
  if col in cat_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
  elif col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())
    all_data[col] = (all_data[col] - all_data[col].mean())/all_data[col].std()

all_data = pd.get_dummies(data = all_data, columns = cat_cols)

train_df = all_data[:ntrain]
test_df = all_data[ntrain:]

# Select main columns to be used in training
main_cols = all_data.columns.difference(['ID', 'target'])
X = train_df[main_cols]
y = train_df.target.astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Optimize model parameters
# I run this code in google colab to make the execution much faster and use the best params in the next code
param_grid = {'min_child_weighth': [1, 5, 10],
        'gamma': [0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 5]
        }
model = GridSearchCV(model, param_grid,n_jobs=-1,verbose=2,cv=5)
model.fit(X_train, y_train)
print(model.best_params_)   

# Make predictions
y_pred = model.predict_proba(X_test)[:, 1]

# Check the auc score of the model
print(f'LGBM AUC score on the X_test is: {roc_auc_score(y_test, y_pred)}\n')

# print classification report
#print(classification_report(y_test, [1 if x >= 0.5 else 0 for x in y_pred]))