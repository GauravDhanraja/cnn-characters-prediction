import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("dataset/labels.csv")

image_folder = "dataset"
img_column = "image"
label_column = "label"
img_size = 128

X, y = [], []

for _, row in df.iterrows():
    img_path = os.path.join(image_folder, row[img_column])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    X.append(image)
    y.append(row[label_column])

X = np.array(X) / 255.0
X = X.reshape(-1, 128, 128, 1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

num_classes = len(le.classes_)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential(
    [
        Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


early_stopping = EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[lr_scheduler, early_stopping],
)

model.summary()
model.save("model.keras")
