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

train = pd.read_csv('./Challenge2/Train.csv')
test = pd.read_csv('./Challenge2/Test.csv')
riders = pd.read_csv('./Challenge2/Riders.csv')
ss = pd.read_csv('./Challenge2/SampleSubmission.csv')

print('Datasets loaded')

# Merge rider dataset to train and test sets
train = train.merge(riders, how = 'left', left_on='rider_id', right_on='Rider ID')
test = test.merge(riders, how = 'left', left_on='rider_id', right_on='Rider ID')

#one-hot encoding
train = pd.get_dummies(train, columns=['client_type', 'vendor_type','order_carrier_type','rider_carrier_type','target'])
test = pd.get_dummies(test, columns=['client_type', 'vendor_type','order_carrier_type','rider_carrier_type'])


#calc euclidean distances
train['rider_to_drop_off_dist'] = (train['rider_lat'] - train['drop_off_lat'])**2 + (train['rider_long'] - train['drop_off_long'])**2
train['rider_to_drop_off_dist'] = train['rider_to_drop_off_dist'].apply(np.sqrt)

train['rider_to_pickup_dist'] = (train['rider_lat'] - train['pickup_lat'])**2 + (train['rider_long'] - train['pickup_long'])**2
train['rider_to_pickup_dist'] = train['rider_to_pickup_dist'].apply(np.sqrt)

test['rider_to_drop_off_dist'] = (test['rider_lat'] - test['drop_off_lat'])**2 + (test['rider_long'] - test['drop_off_long'])**2
test['rider_to_drop_off_dist'] = test['rider_to_drop_off_dist'].apply(np.sqrt)

test['rider_to_pickup_dist'] = (test['rider_lat'] - test['pickup_lat'])**2 + (test['rider_long'] - test['pickup_long'])**2
test['rider_to_pickup_dist'] = test['rider_to_pickup_dist'].apply(np.sqrt)

# Split data
main_cols = train.columns.difference(['ID', 'order_id', 'rider_id', 'Rider ID', 'target_0','target_1','target_2', 'dispatch_time','dispatch_day',	'client_id','drop_off_lat','drop_off_long','pickup_lat','pickup_long','rider_lat','rider_long']).tolist()
#main_cols_test = train.columns.difference(['ID', 'order_id', 'rider_id', 'Rider ID', 'target_0','target_1','target_2', 'dispatch_time','dispatch_day',	'client_id','drop_off_lat','drop_off_long','pickup_lat','pickup_long','rider_lat','rider_long']).tolist()
target_cols = ['target_0','target_1','target_2']
norm_num_cols = ['Active Rider Age', 'Average Partner Rating', 'Number of Ratings','rider_amount','rider_to_drop_off_dist','rider_to_pickup_dist']

#normalize numerical data
train['dispatch_day_of_week'] = train['dispatch_day_of_week']/7
test['dispatch_day_of_week'] = test['dispatch_day_of_week']/7

for col in norm_num_cols:
    #all_data[col] = all_data[col].fillna(all_data[col].median())
    est[col] = (test[col] - test[col].mean())/test[col].std()
    train[col] = (train[col] - train[col].mean())/train[col].std()
    #test[col] = (test[col] - test[col].mean())
    #train[col] = (train[col] - train[col].mean())
    

X = train[main_cols]
y = train[target_cols]

X_test = test[main_cols]

X = X.to_numpy()
y = y.to_numpy()
X_test = X_test.to_numpy()

X = X.astype(np.float32)
y = y.astype(np.float32)
X_test = X_test.astype(np.float32)
model = Sequential()

model.add(Dense(17, input_dim=17, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))


model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) 

history = model.fit(X, y, validation_split=0.1, epochs=20, batch_size=32)

print("Evaluate on test data")
results = model.evaluate(X, y, batch_size=128)
print("test loss, test acc:", results)

prediction = model.predict(X_test)
output = pd.DataFrame(data=prediction)
print(prediction[0:10,0:10])

sub_file = ss.copy()
for x in range(35974): #35974
  #print((output.iloc[x , :]).idxmax())
  sub_file.target[x] = output.iloc[x , :].idxmax()
#sub_file.target = output

sub_file.to_csv('./Challenge2/submission.csv', index = False)

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#cat_cols =['client_type_Business','client_type_Personal','dispatch_day','dispatch_day_of_week','order_carrier_type','order_license_status','rider_carrier_type','rider_license_status','vendor_type_Bike']

#for x in range(19):
#    print(x, X[x].nunique())

