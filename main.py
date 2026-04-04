import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# layers we need, we can add more later if needed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# remap labels to 0-14
from sklearn.preprocessing import LabelEncoder

import numpy as np
import random
import os
import pickle
from pathlib import Path


SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# so our paths work cross platform should
BASE_DIR = Path(__file__).resolve().parent  # folder your script is in

train_path = BASE_DIR / "data" / "train" / "train-70_.pkl"
val_path = BASE_DIR / "data" / "val" / "validation-10_.pkl"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)

with open(val_path, "rb") as f:
    val_data = pickle.load(f)


print("Train keys:", train_data.keys())
print("Val keys:", val_data.keys())

# remap labels from 0-14 pt. 2 > currently random ints such as 163, 28, 6, 189, etc.
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)


X_train = train_data["images"]
y_train = train_data["labels"]
X_val = val_data["images"]
y_val = val_data["labels"]
class_names = train_data["class_names"]

# numpy arrays for easier handling
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# rescaling data from large numbers to numberss from 0 to 1
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32")   / 255.0

print(f"Image shape: {X_train[0].shape}")
print(f"Pixel range: {X_train.min()} - {X_train.max()}")
print(f"Label example: {y_train[0]}")

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")
