#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Adam
from mindspore.communication import init
from mindspore.dataset import GeneratorDataset
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
from datetime import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt



pgf = True
if (pgf):
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


# In[ ]:


# Fix random
seed = 42
reproductibility_mode = True

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
#context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", enable_alltoall=True, device_num=8, global_rank=0)
#init()
np.random.seed(seed)
ms.set_seed(seed)
#context.set_context(save_graphs=True)
#context.set_context(save_graph_path="./graph_exp")


# In[ ]:


# Log folder
log_dir = "logs/mnist/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(log_dir, exist_ok=True)
os.makedirs("figures", exist_ok=True)
print("Logs save in:", log_dir)


# In[ ]:


NUM_CLASSES = 10
input_shape = (28, 28)

shape = input_shape[0] * input_shape[1]


# In[ ]:


# Convert labels to categorical (one-hot encoding)
def to_categorical(y, num_classes=NUM_CLASSES):
    """Convert labels to one-hot encoding"""
    y_categorical = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        y_categorical[i, label] = 1
    return y_categorical


# In[ ]:


def load_mnist(batch_size=64, flatten=True):
    # Load MNIST train and test from mindspore dataset
    mnist_train = ds.MnistDataset(dataset_dir="./MNIST_Data", usage="train", shuffle=True)
    mnist_test = ds.MnistDataset(dataset_dir="./MNIST_Data", usage="test", shuffle=False)

    # Convert to numpy arrays (images and labels)
    x_train, y_train = [], []
    for data in mnist_train.create_dict_iterator(output_numpy=True):
        img = data["image"]  # (32, 32, 3)
        if flatten:
            img = img.reshape(-1)  # vecteur 3072
        x_train.append(img)
        y_train.append(data["label"])

    x_test, y_test = [], []
    for data in mnist_test.create_dict_iterator(output_numpy=True):
        img = data["image"]
        if flatten:
            img = img.reshape(-1)
        x_test.append(img)
        y_test.append(data["label"])

    # Define types
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    # Normaliaation to [0,1]
    x_train /= 255.0
    x_test /= 255.0

    #Transform to one-hot encoding arrays
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    y_dtype = np.float32

    return x_train, y_train, x_test, y_test


# In[ ]:


#HYPERPARAMETERS
hidden_units_large = [512, 512]
hidden_units_med = [256, 256]
hidden_units_low = [64, 64]
hidden_units_verylow = [32, 32]

epochs = 30
input_shape = (28, 28)
dropout = [0.25, 0.25, 0.25]
batch_size = 128


# In[ ]:


x_train, y_train, x_test, y_test = load_mnist(batch_size = batch_size)

print("x_train:", x_train.shape, x_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("x_test:", x_test.shape, x_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)


# In[ ]:


# Define dataset generator
class DatasetGenerator:
    def __init__(self, x, y, batch_size):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.num_samples = len(x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.num_samples

def create_dataset(x, y, batch_size, shuffle=True):
    """Create MindSpore dataset from numpy arrays"""
    dataset_generator = DatasetGenerator(x, y, batch_size)
    dataset = GeneratorDataset(dataset_generator, ["data", "label"], shuffle=shuffle)
    dataset = dataset.batch(batch_size)
    return dataset


# In[ ]:


# Define neural network model
class MLPNet(nn.Cell):
    def __init__(self, input_size, hidden_units, dropout_rates, num_classes):
        super(MLPNet, self).__init__()
        self.layers = nn.SequentialCell()
        
        # First layer
        print("nn.Dense(input_size, hidden_units[0]) = (" +str(input_size) + " " + str(hidden_units[0]) + ")")
        self.layers.append(nn.Dense(input_size, hidden_units[0]))
        self.layers.append(nn.ReLU())
        if dropout_rates[0] > 0:
            self.layers.append(nn.Dropout(keep_prob = 1 - dropout_rates[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_units)):
            print("nn.Dense(hidden_units[i-1], hidden_units[i] with i=" + str(i) + ") = (" +str(hidden_units[i-1]) + " " + str(hidden_units[i]) + ")")
            self.layers.append(nn.Dense(hidden_units[i-1], hidden_units[i]))
            self.layers.append(nn.ReLU())
            if dropout_rates[i] > 0:
                self.layers.append(nn.Dropout(keep_prob = 1 - dropout_rates[i]))
        
        # Output layer
        print("nn.Dense(hidden_units[-1], num_classes) = (" +str(hidden_units[-1]) + " " + str(num_classes) + ")")
        self.layers.append(nn.Dense(hidden_units[-1], num_classes))
    
    def construct(self, x):
        x = x.astype(mstype.float32)
        return self.layers(x)


# In[ ]:


# Training function
def train_and_evaluate_model(x_train, y_train, x_test, y_test, hidden_units_config, model_name):
    print(f"\nTraining {model_name}...")
    
    # Create model
    input_size = x_train.shape[1]
    model = MLPNet(input_size, hidden_units_config, dropout, NUM_CLASSES)
    
    # Define loss and optimizer
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    optimizer = Adam(model.trainable_params(), learning_rate=0.001)
    
    # Create datasets
    train_dataset = create_dataset(x_train, y_train, batch_size, shuffle=True)
    test_dataset = create_dataset(x_test, y_test, batch_size, shuffle=False)
    
    # Define metrics
    accuracy = Accuracy()
    
    # Create model wrapper
    model_wrapper = Model(model, loss_fn, optimizer, metrics={'accuracy': accuracy})
    
    # Training
    start_time = time.time()
    model_wrapper.train(epochs, train_dataset, callbacks=[TimeMonitor(), LossMonitor()])
    train_time = time.time() - start_time
    print(f"---- {train_time} seconds ----")
    
    # Evaluation
    start_time = time.time()
    eval_result = model_wrapper.eval(test_dataset)
    eval_time = time.time() - start_time
    print(f"---- {eval_time} seconds ----")
    
    test_accuracy = eval_result['accuracy']
    print(f"Test accuracy: {test_accuracy}")
    
    # Save model
    ms.save_checkpoint(model, os.path.join(log_dir, f"{model_name}.ckpt"))
    
    return test_accuracy, train_time


# In[ ]:


base_time_train = []

# Train different model sizes
print("Training Standard (Large) Model")
model_large_acc, train_time = train_and_evaluate_model(
    x_train, y_train, x_test, y_test, hidden_units_large, "model_large")
base_time_train.append(train_time)

print("Training Medium Model")
model_med_acc, train_time = train_and_evaluate_model(
    x_train, y_train, x_test, y_test, hidden_units_med, "model_med")
base_time_train.append(train_time)

print("Training Low Model")
model_low_acc, train_time = train_and_evaluate_model(
    x_train, y_train, x_test, y_test, hidden_units_low, "model_low")
base_time_train.append(train_time)

print("Training Very Low Model")
model_verylow_acc, train_time = train_and_evaluate_model(
    x_train, y_train, x_test, y_test, hidden_units_verylow, "model_verylow")
base_time_train.append(train_time)


# In[ ]:


def readMTX(input_path):
    is_init = False
    count = 0
    expected = 0
    with open(input_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            list_line = line.split()
            if list_line[0].isdigit():
                if(not is_init):
                    is_init = True
                    ev = np.zeros((int(list_line[0]), int(list_line[1])))
                    expected = int(list_line[2])
                else:
                    ev[int(list_line[0]) - 1][int(list_line[1]) - 1] = float(list_line[2])
                    count = count + 1
                
    if(expected != count):
        print("Err: Not the same nnz between expected and find ", str(count), "/", str(expected))
    return ev


# In[ ]:


MAX_EV_EXPLORE = 50

acc_test_large = []
acc_test_med = []
acc_test_low = []
acc_test_verylow = []
emb_time_train = []


# In[ ]:


for i in range(MAX_EV_EXPLORE):
    path_ev = "../../data/output/mnist/eigen_" + str(i + 1) + ".mtx"
    try:
        ev = readMTX(path_ev)
        print(f"Eigenvector {i+1} shape: {ev.shape}")
        
        # Apply embedding transformation
        x_train_emb = np.matmul(x_train, ev).astype(np.float32)
        x_test_emb = np.matmul(x_test, ev).astype(np.float32)
        
        # Train models with embeddings
        print(f"Training embedding {i+1} - Large model")
        acc_emb, train_time = train_and_evaluate_model(
            x_train_emb, y_train, x_test_emb, y_test, hidden_units_large, f"emb_{i}_model_large")
        acc_test_large.append(acc_emb)
        emb_time_train.append(train_time)
        
        print(f"Training embedding {i+1} - Medium model")
        acc_emb, train_time = train_and_evaluate_model(
            x_train_emb, y_train, x_test_emb, y_test, hidden_units_med, f"emb_{i}_model_med")
        acc_test_med.append(acc_emb)
        emb_time_train.append(train_time)
        
        print(f"Training embedding {i+1} - Low model")
        acc_emb, train_time = train_and_evaluate_model(
            x_train_emb, y_train, x_test_emb, y_test, hidden_units_low, f"emb_{i}_model_low")
        acc_test_low.append(acc_emb)
        emb_time_train.append(train_time)
        
        print(f"Training embedding {i+1} - Very Low model")
        acc_emb, train_time = train_and_evaluate_model(
            x_train_emb, y_train, x_test_emb, y_test, hidden_units_verylow, f"emb_{i}_model_verylow")
        acc_test_verylow.append(acc_emb)
        emb_time_train.append(train_time)
        
    except Exception as e:
        print(f"Error processing eigenvector {i+1}: {e}")
        continue


# In[ ]:


print("\nFinal Results:")
print("Large model accuracies:", acc_test_large)
print("Medium model accuracies:", acc_test_med)
print("Low model accuracies:", acc_test_low)
print("Very low model accuracies:", acc_test_verylow)


# In[ ]:


#Save results

with open(os.path.join(log_dir, "emb_acc.pkl"), "wb") as f:
    pickle.dump([acc_test_large, acc_test_med, acc_test_low, acc_test_verylow], f)

with open(os.path.join(log_dir, "base_acc.pkl"), "wb") as f:
    pickle.dump([model_large_acc, model_med_acc, model_low_acc, model_verylow_acc], f)

with open(os.path.join(log_dir, "train_time.pkl"), "wb") as f:
    pickle.dump([base_time_train, emb_time_train], f)

