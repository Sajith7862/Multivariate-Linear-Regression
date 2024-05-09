# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1 : Import necessary libraries:

### Step2 : Load the dataset:

### Step3 : Prepare the features (X) and target variable (y):

### Step4 : Split the dataset into training and testing sets:

### Step5 : Train the linear regression model, predict CO2 emissions, and calculate mean squared error:

## Program:
```
DEVELOPED BY MOHAMED HAMEEM SAJITH J
REGISTER NUMBER : 212223240090
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

car_data = pd.read_csv('/content/car.csv')


X = car_data[['Volume', 'Weight', 'CO2']]
y = car_data['CO2']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

sample_data = [[30, 40000, 100]]  
predicted_CO2 = model.predict(sample_data)
print("Predicted CO2 Emission for sample data:", predicted_CO2)


```
## Output:

### Insert your output

![image](https://github.com/Sajith7862/Multivariate-Linear-Regression/assets/145972360/8e66dac5-0a5a-47d3-a5cd-43612c7df7a8)


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
