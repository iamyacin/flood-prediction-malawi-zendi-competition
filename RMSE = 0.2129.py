import pandas as pd
import numpy as np

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

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

target_2019 = regressor.predict(X_2019)

target_2019 = regressor.predict(malawi_2019.iloc[:,1:].values)
Submission = malawi.iloc[:, 39].to_frame()
Submission['target_2019'] = target_2019

Submission['target_2019'].loc[Submission['target_2019'] < 0] = 0.0
Submission['target_2019'].loc[Submission['target_2019'] > 1] = 1.0
Submission.to_csv("Submissions/Submission.csv", index= False)


