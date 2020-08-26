import gc
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2


def rgb_to_grayscale(img_paths, batch_size=15000):
    # get the total number of images
    num_of_imgs = img_paths.shape[0]
    # initialize counter that keeps track of position of image being loaded
    pos = 0
    # initialize empty array in order fill in the image values
    grid = np.zeros((num_of_imgs * 2500, 3))

    for img_path in tqdm(img_paths, total=num_of_imgs):
        # Read the image into a numpy array
        img = cv2.imread(img_path)
        # reshape the image to such that the rgb values are the columns of the matrix
        img = img.reshape(-1, 3)
        # replace the empty array with the values inside the image
        grid[pos: pos + img.shape[0], :] = img
        # update position counter
        pos += img.shape[0]

    # initialize pca to reduce rgb scale to a single dimensional scale
    ipca = IncrementalPCA(n_components=1, batch_size=batch_size)
    # fit pca object to the contents within the grid
    ipca.fit(grid)
    # delete grid to free up sum memory
    del grid
    gc.collect()

    return ipca

def apply_rgb_pca(img,pca_object):
    # reshape image so that pca be applied on the rgb dimension
    img = img.reshape(-1, 3)
    # apply PCA on image
    img = pca_object.transform(img)
    # flatten the image so that it can be stored in a numpy array of images
    img = img.flatten()
    return img


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

for i in tqdm(range(0, 16)):
    if(i!=11):
        print(i)