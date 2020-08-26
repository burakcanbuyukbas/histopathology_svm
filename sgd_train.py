from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


#X_train = np.load("data/nopca/X_train.npy")
Y_train = np.load("data/nopca/Y_train.npy")

#svc = SVC(kernel='rbf', gamma='auto', verbose=True, max_iter=1000000)
clf = SGDClassifier(learning_rate='constant', eta0=0.1, shuffle=True)

print("Step 1:")

x1 = np.load("data_batches/X0.npy", mmap_mode='r')
y1 = Y_train[0:30000]
clf.partial_fit(x1, y1, classes=[0, 1])
del x1, y1
print("Step 2:")

x2 = np.load("data_batches/X1.npy", mmap_mode='r')
y2 = Y_train[30000:60000]
clf.partial_fit(x2, y2, classes=[0, 1])
del x2, y2

print("Step 3:")

x3 = np.load("data_batches/X2.npy", mmap_mode='r')
y3 = Y_train[60000:90000]
clf.partial_fit(x3, y3, classes=[0, 1])
del x3, y3

print("Step 4:")

x4 = np.load("data_batches/X3.npy", mmap_mode='r')
y4 = Y_train[90000:120000]
clf.partial_fit(x4, y4, classes=[0, 1])
del x4, y4

print("Step 5:")

x5 = np.load("data_batches/X4.npy", mmap_mode='r')
y5 = Y_train[120000:150000]
clf.partial_fit(x5, y5, classes=[0, 1])
del x5, y5

print("Step 6:")

x6 = np.load("data_batches/X5.npy", mmap_mode='r')
y6 = Y_train[150000:180000]
clf.partial_fit(x6, y6, classes=[0, 1])
del x6, y6

print("Step 7:")

x7 = np.load("data_batches/X6.npy", mmap_mode='r')
y7 = Y_train[180000:210000]
clf.partial_fit(x7, y7, classes=[0, 1])
del x7, y7

#print(clf.score(X_train, Y_train))

# save the classifier
with open('model2.pkl', 'wb') as savedmodel:
    pickle.dump(clf, savedmodel)

x_test = np.load("data/X_test.npy")
size = x_test.shape[0]
x_test = x_test.flatten().reshape(size, 7500)
y_test = np.load("data/Y_test.npy")
print(clf.score(x_test, y_test))


tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')