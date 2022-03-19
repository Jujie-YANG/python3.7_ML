import pandas as pd
import numpy as np
import sklearn
import openpyxl
from sklearn import linear_model

data = pd.read_excel('./ML_task1.xlsx', header=2)
# data = data[["Student mark out of 100", "Student score"]]
df = pd.DataFrame(data, columns=["Student mark out of 100","Student score"])
# print(data.head())
predict = "Student score"
X = df.drop(predict,1) # Features
y = data[predict] # Labels

# print("X",X,"y",y)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
linear = linear_model.LinearRegression()

# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)
linear.fit(X,y)
print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)
prediction = linear.coef_*66+linear.intercept_
print(int(prediction))