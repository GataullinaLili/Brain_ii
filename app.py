import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Детектор опухоли мозга", layout="centered")

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("🧠 Детектор опухоли на МРТ")
st.markdown("Загрузите изображение МРТ — модель определит наличие/отсутствие опухоли.")

uploaded_file = st.file_uploader("Выберите изображение (.jpg)", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    with st.spinner("Анализируем изображение..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        label = "Опухоль найдена" if prediction > 0.5 else "Опухоль не найдена"
        confidence = f"{prediction * 100:.2f}% уверенность"

    st.success(label)
    st.info(confidence)