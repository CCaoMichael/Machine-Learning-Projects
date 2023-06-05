from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = datasets.load_breast_cancer()

# scale bc scientific notation and big gap between the data
X = scale(bc.data)
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 6)
model = KMeans(n_clusters=2, random_state=0)
model.fit(X_train)

predictions = model.predict(X_test)
labels = model.labels_

print('labels', labels)
print('prediction', predictions)
print('accuracy', accuracy_score(y_test, predictions))

# should read 0 1. if 1 0, flip the predictions
print(pd.crosstab(y_train, labels))