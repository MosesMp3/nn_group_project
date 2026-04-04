import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

import numpy as np
import random
import pickle

from pathlib import Path

from model import moses_model

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR / "data" / "train" / "train-70_.pkl"
val_path = BASE_DIR / "data" / "val" / "validation-10_.pkl"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)

with open(val_path, "rb") as f:
    val_data = pickle.load(f)

X_train = np.array(train_data["images"]).astype("float32")
y_train = np.array(train_data["labels"])
X_val = np.array(val_data["images"]).astype("float32")
y_val = np.array(val_data["labels"])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

print(f"Image shape: {X_train[0].shape}")
print(f"Pixel range: {X_train.min()} - {X_train.max()}")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# build model from class — no duplicate Sequential block
classifier = moses_model()
model = classifier.model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# callbacks
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop]
)

# save weights
model.save_weights("model.pth")
print("Weights saved to model.pth")