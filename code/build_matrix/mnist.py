import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from utils import compute_covariance_matrix
import pickle

from iofiles import save_array_as_mtx, read_mtx



nb_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_vtrain = x_train.reshape(60000, 784)
x_vtest = x_test.reshape(10000,784)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_vtrain = x_vtrain.astype("float32") / 255
x_vtest = x_vtest.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

##Save dataset as mtx file
path_output = "data/training/train_mnist.mtx"
save_array_as_mtx(x_vtrain, path_output)