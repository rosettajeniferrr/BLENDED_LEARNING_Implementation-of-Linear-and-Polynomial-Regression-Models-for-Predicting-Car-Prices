# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Load the dataset using pandas.
2.  Select input features and target variable (price).
3.  Split the data into training and testing sets.
4.  Train the Linear Regression model with scaling.
5.  Train the Polynomial Regression model (degree 2).
6.  Predict prices using both models.
7.  Evaluate performance using MSE, MAE, and R² score and compare results.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Rosetta Jenifer C
RegisterNumber: 212225230230
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

lr=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train,Y_train)
y_pred_linear=lr.predict(X_test)

poly_model=Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression()) 
])
poly_model.fit(X_train,Y_train)
y_pred_poly=poly_model.predict(X_test)

print('Name: Rosetta Jenifer C')
print('Reg no: 212225230230')
print('Linear Regression:')
print('MSE=',mean_squared_error(Y_test,y_pred_linear))
print('MAE=',mean_absolute_error(Y_test,y_pred_linear))
print('R2 Score=',r2_score(Y_test,y_pred_linear))

print("\nPolynomial Regression:")
print("MSE=",mean_squared_error(Y_test,y_pred_poly))
print('MAE=',mean_absolute_error(Y_test,y_pred_poly))
print(f"R2 Score= {r2_score(Y_test,y_pred_poly):.2f}")

plt.figure(figsize=(10,5))
plt.scatter(Y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(Y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()],'r--',label= 'Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
<img width="687" height="286" alt="image" src="https://github.com/user-attachments/assets/41174081-8cd7-49bb-8d78-8bc1c1b36695" />
<img width="1403" height="681" alt="image" src="https://github.com/user-attachments/assets/14e779e1-7bd0-408b-a8d4-b1117bf888e0" />




## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
