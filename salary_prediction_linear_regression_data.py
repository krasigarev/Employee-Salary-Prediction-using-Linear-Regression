# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Data
dataset = pd.read_csv('Salary_Data.csv', engine='pyarrow')
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset omto training and test set
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)

# Train the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set results
y_pred = regressor.predict(x_test)

# Visualize the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualize the test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Predict the salary of an employee with 5 years of experience
mid_level = round(regressor.predict([[5]])[0], 2)
print("Salary prediction for a mid level role:", mid_level)

# Predict the salary of an employee with 12 years of experience
senior_level = round(regressor.predict([[12]])[0], 2)
print("Salary prediction for a senior level role:", senior_level)

# Get the coefficients and intercept
print(round(regressor.coef_[0]), 2)
print(round(regressor.intercept_), 2)