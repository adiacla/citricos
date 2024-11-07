import os
import random
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Configuración de la página con título, ícono, y estado de la barra lateral
st.set_page_config(
    page_title="Detección de enfermedades de los cítricos",
    page_icon=":lemon:",
    initial_sidebar_state='auto'
)

# Ocultar parte del código CSS para personalizar el diseño
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Cargar el modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('citrus_model.keras')  # Ruta del modelo
    return model

with st.spinner('Cargando el modelo...'):
    model = load_model()

# Clases de enfermedades de los cítricos (actualizadas)
class_names = ["Mancha Negra", "Cancro", "Enverdecimiento", "Saludable"]

# Función para predecir la clase de la imagen
def prediction_cls(prediction):
    return class_names[np.argmax(prediction)]  # Devuelve la clase con mayor probabilidad

# Función para importar y predecir la imagen
def import_and_predict(image_data, model):
    # Redimensionar la imagen a 256x256
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Asegurarse de que la imagen esté en formato RGB
    image = image.convert('RGB')
    
    # Convertir la imagen a un array de NumPy
    img = np.asarray(image)
    
    # Normalizar la imagen a [0, 1]
    img = img.astype("float32") / 255.0
    
    # Redimensionar la imagen para que sea compatible con el modelo
    img_reshape = img[np.newaxis, ...]  # (1, 256, 256, 3)

    # Realizar la predicción
    prediction = model.predict(img_reshape)
    return prediction

# Barra lateral
with st.sidebar:
    st.image('hojas_citricos.jpg')  # Imagen de hojas de cítricos
    st.title("Estado de salud de los cítricos")
    st.subheader("Detección de enfermedades en hojas de cítricos usando Deep Learning CNN.")

st.image('Logo_SmartRegions.gif')  # Logo de la empresa
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de enfermedades de los cítricos y su recomendación de tratamiento
         """
         )

# Subir archivo de imagen
file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Por favor cargue una imagen")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Realizar predicción
    predictions = import_and_predict(image, model)

    # Precisión de la predicción (se puede obtener la probabilidad de la clase)
    confidence = np.max(predictions) * 100  # Confianza en la predicción
    st.sidebar.error(f"Precisión: {confidence:.2f}%")

    # Obtener el nombre de la clase predicha
    predicted_class = prediction_cls(predictions)
    st.write(f"Enfermedad detectada: {predicted_class}")

    # Mostrar recomendaciones basadas en la enfermedad predicha
    if predicted_class == 'Saludable':
        st.balloons()
        st.sidebar.success(f"{predicted_class} - La planta está sana.")

    elif predicted_class == 'Mancha Negra':
        st.sidebar.warning(f"Enfermedad detectada: {predicted_class}")
        st.markdown("## Recomendación")
        st.info("La mancha negra se puede controlar mediante tratamientos fungicidas como el azufre o el clorotalonil.")
        st.link_button("Ir a Agrosavia para tratamiento de Mancha Negra", "https://www.agrosavia.co/...")

    elif predicted_class == 'Cancro':
        st.sidebar.warning(f"Enfermedad detectada: {predicted_class}")
        st.markdown("## Recomendación")
        st.info("Podar las ramas infectadas y aplicar productos a base de cobre o fungicidas sistémicos para controlar el cancro.")
        st.link_button("Ir a Agrosavia para tratamiento de Cancro", "https://www.agrosavia.co/...")

    elif predicted_class == 'Enverdecimiento':
        st.sidebar.warning(f"Enfermedad detectada: {predicted_class}")
        st.markdown("## Recomendación")
        st.info("El enverdecimiento puede ser tratado con fungicidas preventivos y asegurando condiciones óptimas de riego y nutrientes.")
        st.link_button("Ir a Agrosavia para tratamiento de Enverdecimiento", "https://www.agrosavia.co/...")
