import pandas as pd

malawi = pd.read_csv('Data/Train.csv')

malawi_class = malawi.copy()

malawi_class['target_2015'].loc[malawi_class['target_2015'] <0.5] = 0.0
malawi_class['target_2015'].loc[malawi_class['target_2015'] >0.5] = 1.0

malawi_land = malawi.iloc[:, 39].to_frame()
malawi_land['X']  = malawi.iloc[:, 0].values
malawi_land['Y'] = malawi.iloc[:, 1].values
malawi_land['elevation']  = malawi.iloc[:, 3]
malawi_land['LC_Type1_mode']  = malawi.iloc[:, 38]

malawi_2015 = pd.concat([malawi_land, malawi.iloc[:, 4:21]], axis = 1, sort = False)
malawi_2019 = pd.concat([malawi_land, malawi.iloc[:, 21:38]], axis = 1, sort = False)

target_2015 = malawi.iloc[:, 2]
target_2015_class = malawi_class.iloc[:, 2]

X = malawi_2015.iloc[:,1:].values
y = target_2015.values
y_class = target_2015_class.values

X_2019 = malawi_2019.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size = 0.05, random_state = 1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_class, y_train_class)

import numpy as np

y15_pred = classifier.predict_proba(X_test_class)
y15_pred = np.delete(y15_pred, 0, 1)

from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y15_pred, y_test))

target_2019 = classifier.predict_proba(X_2019)
target_2019 = np.delete(target_2019, 1, 1)
target_2019 = np.around(target_2019, decimals=3)

Submission = malawi.iloc[:, 39].to_frame()
Submission['target_2019'] = target_2019

Submission['target_2019'].loc[Submission['target_2019'] <0.10] = 0.0
Submission['target_2019'].loc[Submission['target_2019'] >0.90] = 1.0
Submission.to_csv("Submissions/Submission.csv", index= False)

