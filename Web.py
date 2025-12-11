# Web.py — Streamlit CIFAR-10 Image Classifier (Clean + Label Names)
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt


model = load_model("cifar10_model.h5")


class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image (32×32 or larger) and the model will predict its class.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)


    st.success(f"✅ **Predicted Class:** {class_names[predicted_class].capitalize()}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Class Probabilities")
    plt.ylabel("Probability")
    st.pyplot(fig)
