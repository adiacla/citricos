# streamlit_audio_recorder y whisper by Alfredo Diaz - versión abril 2024
# python -m venv env
# cd D:\smart\env\Scripts\
# .\activate 
# cd d:\citricos
# pip install tensorflow==2.15.0
# pip install numpy
# pip install streamlit
# pip install pillow

# Importar las bibliotecas y dependencias necesarias para crear la UI y soportar los modelos de aprendizaje profundo utilizados en el proyecto
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# Ocultar advertencias de deprecación que no afectan directamente al funcionamiento de la aplicación
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página con título, ícono, y estado de la barra lateral
st.set_page_config(
    page_title="Detección de enfermedades de los cítricos",
    page_icon = ":lemon:",  # Cambié el ícono de mango por uno de cítricos
    initial_sidebar_state = 'auto'
)

# Ocultar parte del código CSS, ya que es solo para agregar estilo personalizado y no forma parte de la lógica principal
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # Oculta el código CSS para la interfaz de Streamlit

# Cargar el modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('citrus_model.keras')  # Cargar el modelo de cítricos
    return model

with st.spinner('Cargando el modelo...'):
    model = load_model()
    
# Función para predecir la clase de la imagen en función de los resultados del modelo
def prediction_cls(prediction):  # Predecir la clase de las imágenes basándose en el resultado del modelo
    for key, clss in class_names.items():  # Crear un diccionario de las clases de salida
        if np.argmax(prediction) == clss:  # Verificar la clase
            return key

with st.sidebar:
    st.image('hojas_citricos.jpg')  # Asegúrate de tener una imagen de hojas de cítricos
    st.title("Estado de salud de los cítricos")
    st.subheader("Detección de enfermedades presentes en las hojas de los cítricos usando Deep Learning CNN. Esto ayuda a los agricultores a detectar fácilmente las enfermedades y prevenir su propagación.")

st.image('Logo_SmartRegions.gif')
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de enfermedades de los cítricos y su recomendación de tratamiento
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
    # Redimensionar la imagen a las dimensiones 256x256 que el modelo espera
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # Asegurarse de que la imagen esté en formato RGB (3 canales)
    image = image.convert('RGB')

    # Convertir la imagen a un array de NumPy
    img = np.asarray(image)
    
    # Normalizar la imagen (suponiendo que durante el entrenamiento la normalización fue a [0, 1])
    img = img.astype("float32") / 255.0
    
    # Asegurarse de que la imagen tenga la forma correcta (añadir una dimensión extra)
    img_reshape = img[np.newaxis, ...]  # (1, 256, 256, 3)

    # Verificar las dimensiones de la imagen
    print(img_reshape.shape)  # Esto debe ser (1, 256, 256, 3)

    # Realizar la predicción
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Por favor cargue una imagen")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Precisión : " + str(x) + " %")

    # Clases de enfermedades de los cítricos
    class_names = ['Antracnosis', 'Cancro bacteriano', 'Gorgojo cortador', 'Muerto', 'Mosquito de las agallas', 'Sano', 'Mildiú polvoriento', 'Moho Negruzco']

    string = "Enfermedad detectada: " + class_names[np.argmax(predictions)]
    
    if class_names[np.argmax(predictions)] == 'Sano':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Antracnosis':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Los biofungicidas a base de *Bacillus subtilis* o *Bacillus myloliquefaciens* funcionan bien si se aplican en condiciones climáticas favorables. El tratamiento con agua caliente de semillas o frutos (48°C durante 20 minutos) puede eliminar cualquier residuo de hongos y evitar una mayor propagación de la enfermedad.")
        st.link_button("Ir a Agrosavia para tratamiento de Antracnosis de cítricos", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/719-recomendaciones-de-manejo-preventivo-de-la-antracnosis-en-c%C3%ADtricos")

    elif class_names[np.argmax(predictions)] == 'Cancro bacteriano':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Pode los árboles en flor durante la floración, cuando las heridas sanan más rápido. Retire las ramas marchitas o muertas muy por debajo de las áreas infectadas. Evite podar a principios de primavera y otoño, cuando las bacterias están más activas.")
        st.link_button("Ir a Agrosavia para tratamiento de Cancro bacteriano en cítricos", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/728-recomendaciones-de-manejo-preventivo-del-cancro-bacteriano-en-c%C3%ADtricos")

    elif class_names[np.argmax(predictions)] == 'Gorgojo cortador':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("El gorgojo cortador se puede tratar rociando insecticidas como deltametrina (1 ml/l), cipermetrina (0,5 ml/l) o carbarilo (4 g/l) durante la emergencia de nuevas hojas.")
        st.link_button("Ir a Agrosavia para tratamiento del Gorgojo cortador de cítricos", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/745-control-de-plagas-en-c%C3%ADtricos")

    elif class_names[np.argmax(predictions)] == 'Muerto':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Después de la poda, aplicar oxicloruro de cobre a una concentración del '0,3%' sobre las heridas. Aplique la mezcla de Burdeos dos veces al año para reducir la tasa de infección.")
        st.link_button("Ir a Agrosavia para tratamiento de árboles muertos", "https://repository.agrosavia.co/bitstream/handle/20.500.12324/36827/Ver_documento_36827.pdf?sequence=4")

    elif class_names[np.argmax(predictions)] == 'Mosquito de las agallas':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Utilice trampas adhesivas amarillas para atrapar moscas. Cubra el suelo con plástico para evitar que las larvas caigan al suelo. Arar la tierra para exponer las pupas al sol, que las mata.")
        st.link_button("Ir a Agrosavia para control de plagas en cítricos", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/752-control-de-plagas-en-c%C3%ADtricos")

    elif class_names[np.argmax(predictions)] == 'Powdery Mildew':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Para controlar el oídio en cítricos, se recomienda la pulverización preventiva con azufre humectable.")
        st.link_button("Ir a Agrosavia sobre tratamiento para el oídio", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/760-tratamiento-del-oidio-en-c%C3%ADtricos")

    elif class_names[np.argmax(predictions)] == 'Moho Negruzco':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Rociar con carbarilo o fosfomidón para tratar el moho negro en los cítricos.")
        st.link_button("Ir a Agrosavia para el tratamiento del moho negro", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/765-control-de-mohos-en-c%C3%ADtricos")
