
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Loading the trained model
model = tf.keras.models.load_model('trained_model.h5')
print("Model loaded successfully.")


def ld_imgs(img_path, img_size=(28, 28)):
    img = Image.open(img_path).convert('L')  
    img = img.resize(img_size)  # Resize to 28x28
    img_array = np.array(img)  
    img_array = img_array / 255.0 
    img_array = img_array.reshape(1, 28, 28, 1)  
    return img_array

Sneaker or Ankle Boot
def predict_img(model, img_array):
    prediction = model.predict(img_array)
    return "Ankle Boot" if prediction[0] > 0.5 else "Sneaker"

image_paths = [
    'images/img1.png', 
    'images/img2.png', 
    'images/img3.png', 
    'images/img4.png', 
    'images/img5.png'
]

# Loop to testing
for img_path in image_paths:
    img_array = ld_imgs(img_path)
    predicted_class = predict_img(model, img_array)
    img = Image.open(img_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

