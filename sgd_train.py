from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

X_train = np.load("data/nopca/X_train.npy")
Y_train = np.load("data/nopca/Y_train.npy")

svc = SVC(kernel='rbf', gamma='auto', verbose=True, max_iter=1000000)
clf = SGDClassifier(learning_rate='constant', eta0=0.1, shuffle=True)

for i in tqdm(range(7)):
    print("Step" + str(i) + ":"  )
    x = np.load("data/nopca/X_train.npy", mmap_mode='r')[i*3000:(i+1)*30000]
    y = Y_train[i*3000:(i+1)*30000]
    clf.partial_fit(x, y, classes=[0, 1])



print(clf.score(X_train, Y_train))

# save the classifier
with open('modelSGD.pkl', 'wb') as savedmodel:
    pickle.dump(clf, savedmodel)

# # load model:
# with open('modelSGD.pkl', 'rb') as fin:
#   clf = pickle.load(fin)

# predict on test set(with data from x_train but not used in training):

x_test = np.load("data/nopca/X_train.npy", mmap_mode='r')[210000:224793]
size = x_test.shape[0]
x_test = x_test.flatten().reshape(size, 7500)
y_test = np.load("data/nopca/Y_train.npy", mmap_mode='r')[210000:224793]
clf.predict(x_test)
print(clf.score(x_test, y_test))


tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=clf.predict(x_test)).ravel()

print(f'training set: true negatives: {tn}')
print(f'training set: true positives: {tp}')
print(f'training set: false negatives: {fn}')
print(f'training set: false positives: {fp}')