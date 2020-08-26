import numpy as np
import cv2
from glob import glob
import fnmatch
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import IncrementalPCA
import gc
import h5py
from tqdm import tqdm
import os
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_PCA_data():
    print("Loading data...")
    X = np.load("X.npy", mmap_mode='r')
    #Y = np.load("Y.npy", mmap_mode='r')
    n = X.shape[0]  # how many rows we have in the dataset

    X = X.flatten().reshape(n, 7500)

    chunk_size = 50  # how many rows we feed to IPCA at a time, the divisor of n
    ipca = IncrementalPCA(n_components=50, batch_size=16)

    for i in range(0, n // chunk_size):
        ipca.partial_fit(X[i * chunk_size: (i + 1) * chunk_size])
    print("Fitted.")
    X = ipca.transform(X)
    print("Transformed.")
    np.save('X.npy', X)

    #fit but transform to images one by one and save them to folders!

def do_incremental_pca(batch=50, components=50, path="data/X.h5", target="data_batches/pca100"):
    h5 = h5py.File(path, 'r')
    data = h5['data']
    i = 0
    n = data.shape[0]  # total size
    chunk_size = batch  # batch size
    ipca = IncrementalPCA(n_components=components, batch_size=batch)
    print("Fitting initialized...")
    for i in tqdm(range(0, n // chunk_size)):
        ipca.partial_fit(data[i * chunk_size: (i + 1) * chunk_size])
    print("Trasformation initialized...")
    for i in tqdm(range(0, n // chunk_size)):
        X_ipca = ipca.transform(data[i * chunk_size: (i + 1) * chunk_size])
        np.save("data_batches/pca100/X" + str(i) + ".npy", X_ipca)

    #np.save('Y.npy', Y)

def do_incremental_pca_on_test(batch=50, components=50, path="data/X.h5", target="data_batches"):
    h5 = h5py.File(path, 'r')
    data = h5['data']
    i = 0
    n = data.shape[0]  # total size
    batch_size = batch  # batch size
    ipca = IncrementalPCA(n_components=components, batch_size=batch)
    print("Fitting initialized...")
    for i in tqdm(range(0, n // batch_size)):
        ipca.partial_fit(data[i * batch_size: (i + 1) * batch_size])
    del data
    testdata = np.load("X.npy", mmap_mode='r')
    testshape = testdata.shape
    n = testdata.shape[0]
    print(testshape)
    testdata = testdata.flatten().reshape(testshape[0], testshape[1] * testshape[2] * testshape[3])
    print("Test trasformation initialized...")
    for i in tqdm(range(0, n // batch_size)):
        X_ipca = ipca.transform(testdata[i * batch_size: (i + 1) * batch_size])
        np.save(target + "/X_test" + str(i) + ".npy", X_ipca)
    del testdata
    valdata = np.load(data_folder + "/X_val.npy")
    valshape = valdata.shape
    n = valdata.shape[0]
    print(valshape)
    valdata = valdata.flatten().reshape(valshape[0], valshape[1] * valshape[2] * valshape[3])
    print("Test trasformation initialized...")
    for i in tqdm(range(0, n // batch_size)):
        X_ipca = ipca.transform(valdata[i * batch_size: (i + 1) * batch_size])
        np.save(target + "/X_val" + str(i) + ".npy", X_ipca)

def concat_data():
    arrays = []
    i = 0
    for i in tqdm(range(2)):
        X_add = np.load("data_batches/scaled/X_test" + str(i) + ".npy")
        arrays.append(X_add)
        print(X_add.shape)
    x = np.concatenate(arrays)
    np.save("X_test.npy", x)
    print(x.shape)

def normalized_pca(data, batch=1000, components=100):
    arrays = []
    k = 0
    n = data.shape[0]  # total size
    chunk_size = batch  # batch size
    ipca = IncrementalPCA(n_components=components, batch_size=batch)
    print("Fitting with data...")
    for i in tqdm(range(0, n // chunk_size)):
        ipca.partial_fit(data[i * chunk_size: (i + 1) * chunk_size] / 255)
    print("Trasformation initialized...")
    for i in tqdm(range(0, n // chunk_size)):
        if(i < 20):
            X_spca = ipca.transform(data[i * chunk_size: (i + 1) * chunk_size] / 255)
            np.save("data_batches/scaled/X_train" + str(i) + ".npy", X_spca)
        else:
            X_spca = ipca.transform(data[i * chunk_size: (i + 1) * chunk_size] / 255)
            np.save("data_batches/scaled/X_test" + str(k) + ".npy", X_spca)
            k = k+1

# X, Y = load_data()
# X_Rest, X_val, Y_Rest, Y_val = train_test_split(X, Y, test_size=0.1, shuffle=False)
# X_train, X_test, Y_train, Y_test = train_test_split(X_Rest, Y_Rest, test_size=0.1, shuffle=False)
# save_data_train_test_val(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, X_val=X_val, Y_val=Y_val)
#npy_to_h5()
#concat_data()

#do_incremental_pca(path="X.h5")



# h5 = h5py.File("X.h5", 'r')
# data = h5['data']
# i = 0
# n = data.shape[0]  # total size
# print(str(data.shape))
# normalized_pca(data, batch=10000, components=100)
#concat_data()


# for i in tqdm(range(0, n // chunk_size)):
#     xpart = (data[i * chunk_size: (i + 1) * chunk_size])
#     np.save("data_batches/X" + str(i) + ".npy", xpart)