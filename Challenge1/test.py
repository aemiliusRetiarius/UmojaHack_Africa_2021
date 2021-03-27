import pandas as pd
import numpy as np

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
samplesubmission = pd.read_csv('SampleSubmission.csv')
variable_definations = pd.read_csv('VariableDefinitions.csv')

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
    all_data[col] = all_data[col].fillna(all_data[col].fillna(9999))

all_data = pd.get_dummies(data = all_data, columns = cat_cols)

train_df = all_data[:ntrain]
test_df = all_data[ntrain:]

X = train_df[:,1:2] + train_df[:,4:]
y = train_df[:,3]

model = Sequential()
model.add(Dense(275, input_dim=16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))