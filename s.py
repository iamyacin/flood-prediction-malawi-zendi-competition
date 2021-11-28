import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

malawi = pd.read_csv('Data/Train.csv')

malawi_land = malawi.iloc[:, 39].to_frame()
malawi_land['X']  = malawi.iloc[:, 0].values
malawi_land['Y'] = malawi.iloc[:, 1].values
malawi_land['elevation']  = malawi.iloc[:, 3]
malawi_land['LC_Type1_mode']  = malawi.iloc[:, 38]

malawi_2015 = pd.concat([malawi_land, malawi.iloc[:, 4:21]], axis = 1, sort = False)
malawi_2019 = pd.concat([malawi_land, malawi.iloc[:, 21:38]], axis = 1, sort = False)
target_2015 = malawi.iloc[:, 2]

X15 = malawi_2015.iloc[:,1:].values
X19 = malawi_2019.iloc[:,1:].values
y = target_2015.values

sc_X15 = StandardScaler()
sc_X19 = StandardScaler()
X15 = sc_X15.fit_transform(X15)
X19 = sc_X15.fit_transform(X19)

knn = KNeighborsRegressor()
reg = LinearRegression()
DT = DecisionTreeRegressor()
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
RF = RandomForestRegressor()

models = [knn, reg, DT, svr_lin, svr_poly, svr_rbf, RF]

def regression(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred)) 

for model in models:
    print(model, ' : ', regression(model, X15, y))
    
    
'''
    
    '''
    