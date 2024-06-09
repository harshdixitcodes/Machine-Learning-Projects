import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#house_price = sklearn.datasets.load_boston()
house_price = pd.read_csv('D:\Machine_Learning_Projects\Project_3\Boston.csv')
print(house_price.head())
print(house_price.shape)

#Check for Missing Values
print(house_price.isnull().sum())

print(house_price.describe())

#Correlation between features 
correlation = house_price.corr()

#heatmap
plt.figure(figsize = (10, 10))
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws= {'size':8}, cmap = 'Reds')

#splitting data
X = house_price.drop(['MEDV'], axis = 1)
Y = house_price['MEDV']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 2)

# Model 
model = XGBRegressor()
model.fit(X_train, Y_train)

training_prediction = model.predict(X_train)

# R squared and mean error
score_1 = metrics.r2_score(Y_train, training_prediction)
print("R squared error: ", score_1)

score_2 = metrics.mean_absolute_error(Y_train, training_prediction)
print("Mean Absolute Error: ", score_2)

plt.scatter(Y_train, training_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predictive price")
plt.title("Price Comparison")
plt.show()

test_prediction = model.predict(X_test)
score_3 = metrics.r2_score(Y_test, test_prediction)
print("Test R square error: ", score_3)

score_4 = metrics.mean_absolute_error(Y_test, test_prediction)
print("Test Mean absolute error: ", score_4)