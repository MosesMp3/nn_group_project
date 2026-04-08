import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


import numpy as np
import random
import pickle

from pathlib import Path

from model import moses_model
import os


from tensorflow.keras.callbacks import LearningRateScheduler
import math


def cosine_schedule(epoch, lr):
    return 0.0001 * 0.5 * (1 + math.cos(math.pi * epoch / 200))


submission_check = False  # change to true only when val pickle is changed
from_scratch = False

lr = 0.001 if from_scratch else 0.0001
# if using classifier move learning rate from 0.001 which is for scratch to 0.0001

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR / "data" / "train" / "train-70_.pkl"
val_path = BASE_DIR / "data" / "val" / "validation-10_.pkl"

with open(train_path, "rb") as f:
    train_data = pickle.load(f)

with open(val_path, "rb") as f:  # we can change this to the file they give us
    val_data = pickle.load(f)

X_train = np.array(train_data["images"]).astype("float32")
y_train = np.array(train_data["labels"])
X_val = np.array(val_data["images"]).astype("float32")
y_val = np.array(val_data["labels"])

# norm
X_train = X_train / 255.0
X_val = X_val / 255.0


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

y_train = tf.keras.utils.to_categorical(y_train, 15)
y_val = tf.keras.utils.to_categorical(y_val, 15)


print(f"Image shape: {X_train[0].shape}")
print(f"Pixel range: {X_train.min()} - {X_train.max()}")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# build model from class — no duplicate Sequential block
classifier = moses_model()
model = classifier.model

# add back in when it works, remove when restarting
classifier.load("model.pth")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # changed the rate
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    # loss="sparse_categorical_crossentropy"
    metrics=["accuracy"],
)

model.summary()
# callbacks
early_stop = EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True
)

# augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
)


# learning rate adjustment
cosine_lr = LearningRateScheduler(cosine_schedule)

checkpoint = ModelCheckpoint(
    "best_model.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)


# train, if just getting the score we dont do this
if not submission_check:
    datagen.fit(X_train)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=200,
        callbacks=[early_stop, cosine_lr, checkpoint],
    )

model.load_weights("best_model.weights.h5")
score = model.evaluate(X_val, y_val)
print(f"Best model - Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")

model.save_weights("model.weights.h5")
os.rename("model.weights.h5", "model.pth")


classifier = moses_model()

if not from_scratch:
    classifier.load("model.pth")

predictions = classifier.predict(val_data["images"])


le = LabelEncoder()
y_true = le.fit_transform(val_data["labels"])

accuracy = np.mean(predictions == y_true)
print(f"Test accuracy: {accuracy:.4f}")
