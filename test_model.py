from sklearn.svm import SVC
import numpy as np
import gc
import pickle
import joblib

print("Loading model...")
#load model from pkl
with open('model1.pkl', 'rb') as fin:
  model = pickle.load(fin)



print("Model loaded. Loading data.")

X_test = np.load("data/nopca/X_test.npy")
Y_test = np.load("data/nopca/Y_test.npy")

size = X_test.shape[0]
X_test = X_test.flatten().reshape(size*7500, 1)

print("Sample size: " + str(len(Y_test)))
print("Data loaded. Predicting.")
# print(model.score(X_test[0:2000], Y_test[0:2000]))
# print(model.score(X_test[2000:4000], Y_test[2000:4000]))
# print(model.score(X_test[4000:6000], Y_test[4000:6000]))
# print(model.score(X_test[6000:8000], Y_test[6000:8000]))
# print(model.score(X_test[8000:10000], Y_test[8000:10000]))
# print(model.score(X_test[10000:12000], Y_test[10000:12000]))
# print(model.score(X_test[12000:14000], Y_test[12000:14000]))
# print(model.score(X_test[14000:16000], Y_test[14000:16000]))
# print(model.score(X_test[16000:18000], Y_test[16000:18000]))
# print(model.score(X_test[18000:20000], Y_test[18000:20000]))
# print(model.score(X_test[20000:22000], Y_test[20000:22000]))
# print(model.score(X_test[22000:24000], Y_test[22000:24000]))
print("0-2000")

print(np.count_nonzero(model.predict(X_test[0:2000]) == 0))
print(np.count_nonzero(Y_test[0:2000] == 0))

print("2000-4000")

print(np.count_nonzero(model.predict(X_test[2000:4000]) == 0))
print(np.count_nonzero(Y_test[2000:4000] == 0))

print("4000-6000")

print(np.count_nonzero(model.predict(X_test[4000:6000]) == 0))
print(np.count_nonzero(Y_test[4000:6000] == 0))

print("6000-8000")

print(np.count_nonzero(model.predict(X_test[6000:8000]) == 0))
print(np.count_nonzero(Y_test[6000:8000] == 0))

print("8000-10000")

print(np.count_nonzero(model.predict(X_test[8000:10000]) == 0))
print(np.count_nonzero(Y_test[8000:10000] == 0))

print("10000-12000")

print(np.count_nonzero(model.predict(X_test[10000:12000]) == 0))
print(np.count_nonzero(Y_test[10000:12000] == 0))

print("12000-14000")

print(np.count_nonzero(model.predict(X_test[12000:14000]) == 0))
print(np.count_nonzero(Y_test[12000:14000] == 0))

print("14000-16000")

print(np.count_nonzero(model.predict(X_test[14000:16000]) == 0))
print(np.count_nonzero(Y_test[14000:16000] == 0))

print("16000-18000")

print(np.count_nonzero(model.predict(X_test[16000:18000]) == 0))
print(np.count_nonzero(Y_test[16000:18000] == 0))

print("18000-20000")

print(np.count_nonzero(model.predict(X_test[18000:20000]) == 0))
print(np.count_nonzero(Y_test[18000:20000] == 0))

print("20000-22000")

print(np.count_nonzero(model.predict(X_test[20000:22000]) == 0))
print(np.count_nonzero(Y_test[20000:22000] == 0))

print("22000-24000")

print(np.count_nonzero(model.predict(X_test[22000:24000]) == 0))
print(np.count_nonzero(Y_test[22000:24000] == 0))