import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# layers we need, we can add more later if needed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")
