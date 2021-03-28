import pandas as pd
import numpy as np
import random

from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
    all_data[col] = all_data[col].fillna(all_data[col].mean())
    all_data[col] = (all_data[col] - all_data[col].mean())/all_data[col].std()

all_data = pd.get_dummies(data = all_data, columns = cat_cols)

train_df = all_data[:ntrain]
test_df = all_data[ntrain:]

train_np = train_df.to_numpy()
test_np = test_df.to_numpy()

test_np = np.delete(test_np,3,1)
X =  train_np[:,1:]
pca = PCA(n_components=150)

pca.fit(X)
plt.plot(pca.explained_variance_ratio_)
plt.show()

pcaPlot = PCA(3)
projectData = pcaPlot.fit_transform(X)

plt.scatter(projectData[:, 0], projectData[:, 1], projectData[:,2])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()