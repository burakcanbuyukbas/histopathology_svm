import gc
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
import h5py


print(np.load("data/nopca/Y_train.npy", mmap_mode='r').shape)

# print(h5py.File("X.h5", 'r')["data"].shape)


# path = r'D:\Users\Burak\histopathology'
# patches = glob(path + '/IDC_regular_ps50_idx5/**/*.png', recursive=True)
# patches = np.array(patches)
# rgb_pca = rgb_to_grayscale(patches)
#
#
# #initialize an array containing zeros to place the images in it
# img_array = np.zeros((patches.shape[0],2500))
# # loop through the images and apply PCA on them
# for i,img_path in tqdm(enumerate(patches),total = patches.shape[0]):
#     img = cv2.imread(img_path)
#     try:
#         img_array[i:img.shape[0]] = apply_rgb_pca(img, rgb_pca)
#     except:
#         pass

