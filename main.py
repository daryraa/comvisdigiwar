# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# === Load model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_final.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Optional
    return model

model = load_model()

# === Class names (urutan harus sama dengan saat training) ===
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 
               'paper', 'plastic', 'shoes', 'trash']

# === UI ===
st.title("Garbage Classification with MobileNetV2")
st.write("Upload gambar sampah untuk dideteksi secara otomatis.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption='Gambar yang diupload', use_column_width=True)

    # Preprocess
    img_resized = image_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    # Tampilkan hasil
    st.markdown(f"### Prediksi: **{pred_class}**")
    st.markdown(f"Confidence: `{confidence * 100:.2f}%`")
