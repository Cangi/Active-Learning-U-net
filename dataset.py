import os
import sys
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from tqdm import tqdm
import warnings



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#Constants for image proccessing 
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

#Paths to the train and test images
TRAIN_PATH = os.path.realpath('') + '/../hon/input/train/'
TEST_PATH = os.path.realpath('') + '/../hon/input/test/'
print(TRAIN_PATH)
#Iterators for the trainning images and for the test images
train_ids = next(os.walk(TRAIN_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]

def get_dataset():
    x_train, y_train = train_data_Preparation()
    x_test, y_test = test_data_Preparation()
    return x_train, y_train, x_test, y_test

def train_data_Preparation():
    images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    #Resizing the images
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        images[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        #Resizing the labels for the images
        for mask_file in next(os.walk(TRAIN_PATH + '/masks/'))[2]:
            mask_ = imread(TRAIN_PATH + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
    labels[n] = mask

    #Assign the resized images and their labels
    X_train = images
    Y_train = labels
    
    print('Done')

    return X_train, Y_train

def test_data_Preparation():
    global X_test
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        print(path)
        img = imread(path)[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(TEST_PATH + '/masks/'))[2]:
            mask_ = imread(TEST_PATH + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
            Y_test[n] = mask

    print('Done!')

    return X_test, Y_test
