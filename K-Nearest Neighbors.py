import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

filepath = 'car.data'
data = pd.read_csv('filepath')
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

X = data[['buying', 'maint', 'safety']].values
y = data[['class']]


Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

label_mapping = {'unacc':10, 'acc':1, 'good':2, 'vgood':3}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('predictions:', prediction)
print('accuracy:', accuracy)

print('actual value:', y[20])
print('predicted value:', knn.predict(X)[20])