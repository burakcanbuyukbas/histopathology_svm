from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


X_train = np.load("X_train.npy")
Y_train = np.load("data/nopca/Y_train.npy")[0:200000]

#svc = SVC(kernel='rbf', gamma='auto', verbose=True, max_iter=1000000)
clf = SGDClassifier(eta0=0.1, shuffle=True)


clf.fit(X_train, Y_train)


#print(clf.score(X_train, Y_train))

# save the classifier
with open('modelSGDNormalized.pkl', 'wb') as savedmodel:
    pickle.dump(clf, savedmodel)

x_test = np.load("X_test.npy")
y_test = np.load("data/nopca/Y_train.npy")[200000:220000]
print(clf.score(x_test, y_test))


tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')