import pandas as pd
data=pd.read_csv("E:\Chrome downloads\ML\ML1\Cars.csv")

X=data.drop(["MPG"],axis=1,inplace=False)
y=data.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_pred=regressor.predict(X)

import numpy as np
from sklearn import metrics
rmse=np.sqrt(metrics.mean_squared_error(y, y_pred))
print(rmse)


X1=np.log(X)
regressor1=LinearRegression()
regressor1.fit(X1,y)
y_pred1=regressor1.predict(X1)
rmse1=np.sqrt(metrics.mean_squared_error(y, y_pred1))
print(rmse1)


y1=np.log(y)
regressor2=LinearRegression()
regressor2.fit(X,y1)
y_pred2=regressor2.predict(X)
y1_actual=np.exp(y_pred2)
rmse2=np.sqrt(metrics.mean_squared_error(y, y1_actual))
print(rmse2)

X2=X*X
regressor3=LinearRegression()
regressor3.fit(X2,y)
y_pred3=regressor3.predict(X2)
rmse3=np.sqrt(metrics.mean_squared_error(y, y_pred3))
print(rmse3)

import pickle
pickle.dump(regressor1,open('model_cars.pkl','wb'))
