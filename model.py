import tensorflow as tf
from tensorflow.keras import layers, models

# Example: simple CNN for 2 classes (cat vs dog)
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")   # 🔹 adjust classes
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train on your dataset
# (X_train, y_train), (X_val, y_val) = ...
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save
model.save("model.h5")
