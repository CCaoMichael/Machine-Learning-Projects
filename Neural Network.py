from PIL import Image
import numpy as np
import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

X_train = mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images()
y_test = mnist.test_labels()

# print('X_train', X_train)
# print('X shape', X_train.shape)

# print('y_train', y_train)
# print('y shape', y_train.shape)
# print(X_train[0])

print(X_train.shape)

X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

X_train =(X_train/256)
X_test = (X_test/256)

# formatting done. can start on model
clf = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (64,64))
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
acc = confusion_matrix(y_test, predictions)


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

print('accuracy: ', accuracy(acc))