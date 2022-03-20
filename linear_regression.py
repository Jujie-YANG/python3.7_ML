import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style # grid style
import pickle


# Predict students' grades

data = pd.read_csv('Linear Regression/student/student-mat.csv', sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
data = shuffle(data) # Optional - shuffle the data
predict = "G3"



'''
# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy", acc)

    if acc > best:
        best = acc
        with open("studentGrades.pickle", "wb") as f:  # opens the file in binary format for writing
            pickle.dump(linear, f) # pickle.dump() save model into our  created pickle file(f)
'''

# Loading Our Model
# Now we have save the best model and then we can use linear to predict grades like before
pickle_in = open("Linear Regression/studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)

X = np.array(data.drop([predict], 1))  # y axis
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
predictions = linear.predict(x_test)
# print(x_train,"\n",y_train)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "failures" # Change this to G1, G2, studytime or absences to see other graphs
style.use("ggplot")
plt.scatter(data[plot], data["G3"])
plt.legend(plot, loc='best')
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()