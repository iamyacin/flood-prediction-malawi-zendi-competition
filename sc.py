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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_2019 = sc.transform(X_2019)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_2019 = pca.transform(X_2019)

explained_variance = pca.explained_variance_ratio_

# model
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.around(y_pred, decimals=3)

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_pred, y_test))

y_19 = model.predict(X_2019)

target_2019 = np.around(y_19, decimals=3)

Submission = malawi.iloc[:, 39].to_frame()
Submission['target_2019'] = target_2019

Submission.to_csv("Submissions/Submission.csv", index= False)



