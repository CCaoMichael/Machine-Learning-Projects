from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# We want to make sure that the y (label) prediction has the same shape as the origin X (figure) data
# print(X.shape)
# print(y.shape)

classes = ['Iris Setosa', 'Iris Versicolor,', 'Iris Virginica']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = svm.SVC()
model.fit(X_train, y_train)
print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print('predictions:', predictions)
print('actual: ', y_test)
print('accuracy:', acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])