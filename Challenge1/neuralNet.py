import pandas as pd
import numpy as np
import random
import tensorflow as tf

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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

train_np = train_df.to_numpy()
test_np = test_df.to_numpy()

test_np = np.delete(test_np,3,1)
pred = test_np[:,1:]
pred= pred.astype(np.float32)

y = train_np[:,3][:,np.newaxis]
train_np = np.delete(train_np,3,1)
X =  train_np[:,1:]


X = X.astype(np.float32)
y = y.astype(np.float32)
model = Sequential()

model.add(Dense(274, input_dim=274, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC']) 

history = model.fit(X, y, validation_split=0.1, epochs=4, batch_size=1024)


prediction = model.predict(pred)
output = pd.DataFrame(data=prediction)
print(prediction[:,0:10])

sub_file = samplesubmission.copy()
sub_file.target = output
sub_file.to_csv('./Challenge1/submission.csv', index = False)

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()