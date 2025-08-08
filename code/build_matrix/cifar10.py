import numpy as np
import pandas as pd
import keras
from keras.datasets import cifar10
from utils import compute_covariance_matrix
import pickle

from iofiles import save_array_as_mtx, read_mtx



NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#Reshape: from (x, 32, 32, 3) to (x, 3072)
x_vtrain = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_vtest = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])


x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


x_vtrain = x_vtrain.astype("float32") / 255
x_vtest = x_vtest.astype("float32") / 255


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

##Save dataset as mtx file
path_output = "data/training/train_cifar10.mtx"
save_array_as_mtx(x_vtrain, path_output)