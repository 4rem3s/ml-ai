
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Loading dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# binary classification: Sneakers (7) and Ankle boots (9)
train_filter = np.where((train_labels == 7) | (train_labels == 9))
test_filter = np.where((test_labels == 7) | (test_labels == 9))

train_images_binary = train_images[train_filter]
train_labels_binary = train_labels[train_filter]
test_images_binary = test_images[test_filter]
test_labels_binary = test_labels[test_filter]

# Sneakers = 0, Ankle boots = 1;
train_labels_binary = np.where(train_labels_binary == 9, 1, 0)
test_labels_binary = np.where(test_labels_binary == 9, 1, 0)

# Normalizing & reshape 
train_images_binary = train_images_binary / 255.0
test_images_binary = test_images_binary / 255.0

train_images_binary = train_images_binary.reshape(-1, 28, 28, 1)
test_images_binary = test_images_binary.reshape(-1, 28, 28, 1)

# CNNet
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # output;
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images_binary, train_labels_binary, epochs=7, 
          validation_data=(test_images_binary, test_labels_binary))

# Saving
model.save('trained_model.h5')
