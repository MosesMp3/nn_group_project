import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense
import numpy as np


class moses_model:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(
            [
                # Block 1
                Conv2D(64, (3, 3), padding="same", input_shape=(64, 64, 3)),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                Conv2D(64, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                # Block 2
                Conv2D(128, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                Conv2D(128, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                MaxPooling2D(2, 2),
                Dropout(0.3),
                # Block 3
                Conv2D(256, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                Conv2D(256, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                MaxPooling2D(2, 2),
                Dropout(0.35),
                # Block 4
                Conv2D(512, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                Conv2D(512, (3, 3), padding="same"),
                BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                Dropout(0.4),
                # Classification head
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(15, activation="softmax"),
            ]
        )
        return model

    def load(self, weights_path="model.pth"):
        import os

        temp_path = weights_path.replace(".pth", ".weights.h5")
        os.rename(weights_path, temp_path)
        self.model.load_weights(temp_path)
        os.rename(temp_path, weights_path)

    def predict(self, images):
        images = np.array(images).astype("float32")
        if images.max() > 1.0:
            images = images / 255.0
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)
