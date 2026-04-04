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

# remap labels from 0-14 pt. 2 > currently random ints such as 163, 28, 6, 189, etc.
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

X_train = X_train.astype("float32")
X_val   = X_val.astype("float32")

print(f"Image shape: {X_train[0].shape}")
print(f"Pixel range: {X_train.min()} - {X_train.max()}")
print(f"Label example: {y_train[0]}")

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

input_shape = (64, 64, 3)
num_classes = 15

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="valid"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="valid"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- TRAIN ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)
