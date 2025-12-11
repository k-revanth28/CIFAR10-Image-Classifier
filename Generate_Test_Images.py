# Generate_Test_Images.py

# This script downloads CIFAR-10 test images and saves 10 random samples
# into a folder named 'test_images' for easy testing with your model or Streamlit app.

import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from PIL import Image

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = cifar10.load_data()

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Create folder for saving images
save_dir = "test_images"
os.makedirs(save_dir, exist_ok=True)

print(f"Saving 10 random CIFAR-10 test images to '{save_dir}'...\n")

# Save 10 random test images
for i in range(10):
    idx = np.random.randint(0, len(x_test))
    img = Image.fromarray(x_test[idx])
    label = class_names[y_test[idx][0]]
    file_path = os.path.join(save_dir, f"test_image_{i+1}_{label}.png")
    img.save(file_path)
    print(f"Saved: {file_path}")

print("\n All test images saved successfully! You can find them in the 'test_images' folder.")
