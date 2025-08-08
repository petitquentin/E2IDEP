# Data folder

This folder contains the datasets used in this project. Please follow the instructions below to download and place the files correctly.

---

## Required datasets

### 1. Date Fruit & Fetal health dataset

- No manual download is needed.
- These datasets are already in the git repository folder.

### 2. Radar dataset (Winnipeg)

- Visit: [UCI Dataset - Crop Mapping Using Fused Optical Radar](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)
- Download and extract: `WinnipegDataset.txt`
- Place the file in this `data/` folder.

The dataset can be directly downloaded with the following command:
```bash
wget https://dainesanalytics.com/datasets/crop-mapping-winnipeg/WinnipegDataset.txt
```

### 3. MNIST & CIFAR-10

- No manual download is needed.
- These datasets are loaded using `keras.datasets` directly in Python.

---

##  Expected Folder Structure

data/ \
├── date_fruit_datasets.csv \
├── fetal_health.csv \
├── WinnipegDataset.txt \

> Ensure all datasets are correctly named and placed as above.

---

## Build matrix

After installing dependencies from `requirements.txt`, you can build the training `.mtx` matrix by running:

```bash
sh scripts/build_matrix.sh
```

All training `.mtx` matrix will be generated in the folder `data/training/`


