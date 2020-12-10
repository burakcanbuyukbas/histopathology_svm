from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import result
from utils import load_test

X_train = np.load("data/nopca/X_train.npy")
print(X_train.shape)
Y_train = np.load("data/nopca/Y_train.npy")
print(Y_train.shape)


X_test, Y_test = load_test()
print(X_test.shape)
print(Y_test.shape)
#.flatten().reshape(size, 7500)

svc = SVC(kernel='rbf', gamma='auto', verbose=True, max_iter=10000000000)
svc.fit(X_train, Y_train)

#print(clf.score(X_train, Y_train))

# save the classifier
with open('model5.pkl', 'wb') as savedmodel:
    pickle.dump(svc, savedmodel)

x_test = np.load("X_test.npy")
y_test = np.load("data/nopca/Y_train.npy")[200000:220000]
print(svc.score(x_test, y_test))


tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=svc.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')

result.plot_graph(svc, x_test, y_test)