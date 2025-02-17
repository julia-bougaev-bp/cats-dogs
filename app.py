import streamlit as st
from fastai.vision.all import *

from pathlib import Path

import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

def is_cat(x):
    return x[0].isupper() 


model_path = "model.pkl"
learn_inf = load_learner(model_path)


st.title("🖼️ Image Classification App")
st.write("Upload an image and let the model classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
 
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = PILImage.create(uploaded_file)
  
    pred, pred_idx, probs = learn_inf.predict(img)
 
    st.write(f"### 🏷️ Prediction: **{pred}**")
    st.write(f"### 📊 Probability: {probs[pred_idx]:.4f}")

    st.bar_chart(probs.numpy())

