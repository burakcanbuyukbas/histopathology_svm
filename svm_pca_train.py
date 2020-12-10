from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import result
from utils import load_test

X_train = np.load("data/pca/100/X_train.npy")
print(X_train.shape)
Y_train = np.load("data/nopca/Y_train.npy")[0:200000]
print(Y_train.shape)




svc = SVC(kernel='rbf', gamma='auto', verbose=True)
svc.fit(X_train, Y_train)

#print(clf.score(X_train, Y_train))

# save the classifier
with open('model5.pkl', 'wb') as savedmodel:
    pickle.dump(svc, savedmodel)


# predict with test set
X_test = np.load("data/pca/100/X_test.npy")
Y_test = np.load("data/nopca/Y_train.npy")[200000:220000]
print(X_test.shape)
print(Y_test.shape)
#.flatten().reshape(size, 7500)


print(svc.score(X_test, Y_test))


tn, fp, fn, tp = confusion_matrix(y_true=X_test, y_pred=svc.predict(X_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')

result.plot_graph(svc, X_test, Y_test)