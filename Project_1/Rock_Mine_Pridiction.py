import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Data Processing
sonar_data = pd.read_csv('D:\Machine_Learning_Projects\Project_1\sonar data.csv', header = None)
sonar_data.head()
sonar_data.shape

sonar_data.describe() #describe statistical measures
sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

#seperating data and labels
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

#training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)
print(X.shape, x_train.shape , x_test.shape)

#Model Traning ---> Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)

#Model Evaluation
X_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print('Accuracy on Training data: ', training_data_accuracy)

X_test_prediction = model.predict(x_test)
test_data_prediction = accuracy_score(X_test_prediction, y_test)

print("Accuracy on test data: ", test_data_prediction)

#Making a Predictive System
input_data = ()
#changing data to array
input_data_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print('The object is a Rock.')
else:
    print('The object is a mine')