import pandas as pd

malawi = pd.read_csv('Data/Train.csv')

malawi_land = malawi.iloc[:, 39].to_frame()
malawi_land['X']  = malawi.iloc[:, 0].values
malawi_land['Y'] = malawi.iloc[:, 1].values
malawi_land['elevation']  = malawi.iloc[:, 3]
malawi_land['LC_Type1_mode']  = malawi.iloc[:, 38]

malawi_2015 = pd.concat([malawi_land, malawi.iloc[:, 4:21]], axis = 1, sort = False)
malawi_2019 = pd.concat([malawi_land, malawi.iloc[:, 21:38]], axis = 1, sort = False)

target_2015 = malawi.iloc[:, 2]

X = malawi_2015.iloc[:,1:].values
y = target_2015.values
X_2019 = malawi_2019.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

from keras.models import Sequential
from keras.layers import Dense

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(21, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(42, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(42, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(42, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()

NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2)
y_pred= NN_model.predict(X_test)

import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_pred, y_test))

target_2019 = NN_model.predict(X_2019)

Submission = malawi.iloc[:, 39].to_frame()
Submission['target_2019'] = target_2019

Submission['target_2019'].loc[Submission['target_2019'] > 1] = 1.0
Submission.to_csv("Submissions/Submission.csv", index= False)

