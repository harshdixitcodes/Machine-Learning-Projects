import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('D:\Machine_Learning_Projects\Project_2\diabetes.csv')
print(diabetes_dataset.shape)

# Statistical Measures
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())

print(diabetes_dataset.groupby('Outcome').mean())

# Seperating the Data and Labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardize_data = scaler.transform(X)
X = standardize_data
Y = diabetes_dataset['Outcome']

# Data spilitting TRAIN and TEST
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training 
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)

#Model Evaluation
X_train_predict = classifier.predict(X_train)
training_accuracy = accuracy_score(X_train_predict, Y_train)
print("Accuracy score of the training data: ", training_accuracy)

X_test_predict = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_predict, Y_test)
print("Accuracy score of the test data: ", test_accuracy)

# Predictive System
input_data = ()
data_array = np.asarray(input_data).reshape(1, -1)
data_array = scaler.transform(data_array)
print(data_array)

prediction = classifier.predict(data_array)
print(prediction)
if (prediction[0] == 0):
    print("The patient is Non-Diabetic")
else:
    print("The patient is Diabetic")

