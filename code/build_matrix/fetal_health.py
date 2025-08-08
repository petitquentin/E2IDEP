import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from iofiles import save_array_as_mtx, read_mtx



NUM_CLASSES = 3
CLASS_COL = "fetal_health"
dataset_path = "./data/fetal_health.csv"
ds_import = pd.read_csv(dataset_path, sep=",", header=0, index_col=False)

total_column_count = len(ds_import.columns)
total_feature_count = total_column_count-1

# We create attribute-only and target-only datasets (df_features_train and df_target_train)
df_y_train = ds_import[CLASS_COL]
df_x_train = ds_import.drop([CLASS_COL], axis=1)

y = keras.utils.to_categorical(np.asarray(df_y_train.factorize()[0]))

#Split dataset

x_train, x_test, y_train, y_test = train_test_split(df_x_train, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
scaler.mean_
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
scaled_data_df = pd.DataFrame(x_train, columns=df_x_train.columns)

##Save dataset as mtx file
path_output = "data/training/train_fetal_health.mtx"
save_array_as_mtx(x_train, path_output)