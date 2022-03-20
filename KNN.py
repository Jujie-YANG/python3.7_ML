import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("KNN/car.data")

le = preprocessing.LabelEncoder()  # label encode each column of our data into integers

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  # optional

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features, zip() - a tuple of 121 tuples ((,),(,))
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
# print(x_train,"\n",y_train)

# Training a KNN Classifier
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# Testing Our Model
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]  # preprocessing - 0, 1, 2, 3

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

    # kneighbors(X=None, n_neighbors=None, return_distance=True)
    n = model.kneighbors([x_test[x]], 9, True)  # [x_test[x]] - have to be two dimension
    print("N: ", n)  # Returns distances and indices of the neighbors of each point.
