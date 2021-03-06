import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()  # Loading Sklearn Datasets
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train, y_train)
clf = svm.SVC(kernel="linear", C=2)  # By default our kernel has a soft margin of value 1
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)