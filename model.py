import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

class moses_model:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Conv2D(64, (3,3), activation="relu", padding="same", input_shape=(64,64,3)),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),

            Conv2D(128, (3,3), activation="relu", padding="valid"),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),

            Conv2D(256, (3,3), activation="relu", padding="valid"),
            BatchNormalization(),
            MaxPooling2D(2,2),

            Conv2D(512, (3,3), activation="relu", padding="valid"),
            BatchNormalization(),
            MaxPooling2D(2,2),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(15, activation="softmax")
        ])
        return model
    
    def load(self, weights_path="model.pth"):
        self.model.load_weights(weights_path)

    def predict(self, images):
        images = np.array(images).astype("float32")
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)