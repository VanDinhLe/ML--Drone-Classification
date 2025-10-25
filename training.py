import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model


data_dir = '../img/data_64'  # Root folder containing 'well-mixed' and 'un-mixed'
img_size = (320, 320)
# batch_size = 32
batch_size = 8

# Load dataset using image_dataset_from_directory
dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/train",
    image_size=img_size,
    batch_size=batch_size,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/valid",
    image_size=img_size,
    batch_size=batch_size,
)

dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# ResUnit building block
def ResUnit(x):
    shortcut = x
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv3x3 = layers.Conv2D(32, (3, 3), padding='same')(conv3x3)
    
    # Add skip connection (not concatenate)
    output = layers.add([conv3x3, shortcut])
    output = layers.ReLU()(output)
    return output

# ResStack block built from ResUnit
def ResStack(x):
    conv1x1 = layers.Conv2D(32, (1, 1), activation='linear', padding='same')(x)

    output = ResUnit(conv1x1)
    output = ResUnit(output)

    return output

# Define DRNN model
def create_model():
    inputs = keras.Input(shape=(320, 320, 3))
    x = layers.Rescaling(1./255)(inputs)  # Normalize input
    x = ResStack(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = ResStack(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResStack(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResStack(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResStack(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResStack(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = layers.Flatten()(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Plot and save model architecture
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# Train model
history = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=10
)

# Evaluate model
loss, acc = model.evaluate(val_dataset)
print(f"Test Accuracy: {acc:.4f}")

# Save the model
model.save('best_model.keras')


# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
