import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalising
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshaping
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# NNet
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Output 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, validation_split=0.1)  

# evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

def test_img(image_folder):
    plt.figure(figsize=(10, 10))
    for image_number in range(1, 4):
        img_path = os.path.join(image_folder, f'img{image_number}.png')

        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L')  
            img = img.resize((28, 28))  
            img_array = np.array(img) / 255.0  
            img_array = np.expand_dims(img_array, axis=-1)  
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            plt.subplot(2, 3, image_number)  
            plt.imshow(img_array[0].squeeze(), cmap='gray') 
            plt.title(f'Predicted: {predicted_class}')  
            plt.axis('off')
        else:
            print(f"File {img_path} does not exist.")

    plt.tight_layout()
    plt.show()

test_img('images')

