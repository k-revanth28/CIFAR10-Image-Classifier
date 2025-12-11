# DP1.py — Utility functions for feature extraction (NO OPENCV VERSION)

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler
from PIL import Image

# ---------------------------------------
# 🔹 Function: Preprocess image (PIL only)
# ---------------------------------------
def preprocess_image(image):
    """
    Takes a PIL image and returns:
    - CNN formatted image (32×32 normalized)
    - KNN features (flattened 3072 values, scaled)
    """

    # Ensure image is RGB
    image = image.convert("RGB")

    # Resize for CNN
    cnn_img = image.resize((32, 32))
    cnn_img = img_to_array(cnn_img) / 255.0
    cnn_img = np.expand_dims(cnn_img, axis=0)

    # Resize for KNN
    knn_img = image.resize((32, 32))
    knn_img = np.array(knn_img).flatten().reshape(1, -1)

    # Standard scaling (important for KNN)
    scaler = StandardScaler()
    knn_img = scaler.fit_transform(knn_img)

    return cnn_img, knn_img
