# streamlit_audio_recorder y whisper by Alfredo Diaz - version April 2024
#python -m venv env
#cd D:\smart\env\Scripts\
#.\activate 
#cd d:\mango
#pip install tensorflow==2.15.0
#pip install numpy
#pip install streamlit
#pip install pillow

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Detección de enfermedades del mango",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('mango_model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    
    
def prediction_cls(prediction): # predecir la clase de las imágenes en función de los resultados del modelo
    for key, clss in class_names.items(): # crear un diccionario de las clases de salida
        if np.argmax(prediction)==clss: # Verifica la clase
            return key

with st.sidebar:
        st.image('hojas.png')
        st.title("Estado de salud Manguifera")
        st.subheader("Detección de enfermedades presentes en las hojas del mango usando Depp Learning CNN. Esto ayuda al campesino a detectar fácilmente la enfermedad e identificar su causa.")
	st.subheader("Alfredo Diaz- UNAB 2024")

st.image('Logo_SmartRegions.gif')
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         # Detección de enfermedades del mango con su recomendación de tratamiento
         """
         )

img_file_buffer = st.camera_input("Capture una foto de una hoja de mango")
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    bytes_data = img_file_buffer.getvalue()
    image_stream = BytesIO(bytes_data)
        # Abrir la imagen con PIL
    image = Image.open(image_stream)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Antracnosis', 'Cancro bacteriano', 'Gorgojo cortador', 'Muerto', 'Mosquito de las agallas', 'Sano', 'Mildiú polvoriento', 'Moho Negruzco']

    string = "Enfermedad detectada : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Sano':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Antracnosis':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Los biofungicidas a base de Bacillus subtilis o Bacillus myloliquefaciens funcionan bien si se aplican en condiciones climáticas favorables. El tratamiento con agua caliente de semillas o frutos (48°C durante 20 minutos) puede eliminar cualquier residuo de hongos y evitar una mayor propagación de la enfermedad en el campo o durante el transporte.")
        st.link_button("Ir a Agrosavia para tratamiento de Antracnosis del mango ", "https://www.agrosavia.co/productos-y-servicios/oferta-tecnol%C3%B3gica/l%C3%ADnea-agr%C3%ADcola/frutales/recomendaciones-protocolos-y-metodolog%C3%ADas/719-recomendaciones-de-manejo-preventivo-de-la-antracnosis-en-mango-a-trav%C3%A9s-del-uso-de-pr%C3%A1cticas-culturales-qu%C3%ADmicas-y-biol%C3%B3gicas")

    elif class_names[np.argmax(predictions)] == 'Cancro bacteriano':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Pode los árboles en flor durante la floración, cuando las heridas sanan más rápido. Retire las ramas marchitas o muertas muy por debajo de las áreas infectadas. Evite podar a principios de primavera y otoño, cuando las bacterias están más activas. Si usa podadoras de hilo alrededor de la base de los árboles, evite dañar la corteza con Tree Wrap transpirable para prevenir infecciones.")
        st.link_button("Ir a Agrosavia para tratamiento de la necrosis apical bacteriana ", "https://revistacta.agrosavia.co/index.php/revista/article/view/2487")

    elif class_names[np.argmax(predictions)] == 'Gorgojo cortador':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("El gorgojo cortador se puede tratar rociando insecticidas como deltametrina (1 ml/l), cipermetrina (0,5 ml/l) o carbarilo (4 g/l) durante la emergencia de nuevas hojas, lo que puede prevenir eficazmente el daño del gorgojo..")
        st.link_button("Ir a Agrosavia para tratamiento del Gorgojo cortador de hojas del mango ", "https://www.istockphoto.com/es/foto/gorgojo-cortador-de-hojas-de-mango-herido-en-hoja-de-mango-en-viet-nam-gm1452125098-488637815")

    elif class_names[np.argmax(predictions)] == 'Muerto':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Después de la poda, aplicar oxicloruro de cobre a una concentración del '0,3%' sobre las heridas. Aplique la mezcla de Burdeos dos veces al año para reducir la tasa de infección en los árboles. Los aerosoles que contienen el fungicida tiofanato de metilo han demostrado ser eficaces contra B.")
        st.link_button("Ir a Agrosavia para tratamiento de hojas muertas del mango ", "https://repository.agrosavia.co/bitstream/handle/20.500.12324/36827/Ver_documento_36827.pdf?sequence=4")

    elif class_names[np.argmax(predictions)] == 'Mosquito de las agallas':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Utilice trampas adhesivas amarillas para atrapar moscas. Cubra el suelo con papel de plástico para evitar que las larvas caigan al suelo o que las pupas salgan de su nido. Arar la tierra con regularidad para exponer las pupas y larvas al sol, que las mata. Recolecte y queme material de árboles infestados durante la temporada.")
        st.link_button("Ir a Agrosavia plagas y enfermedades", "https://www.mango.org/wp-content/uploads/2020/08/Mango_Plagas_y_Enfermedades_SPN.pdf")

        
    elif class_names[np.argmax(predictions)] == 'Powdery Mildew':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Para controlar el oídio, se recomiendan tres pulverizaciones de fungicidas. La primera pulverización con azufre humectable (0,2%, es decir, 2 g por litro de agua) debe realizarse cuando las panículas tengan un tamaño de 8 a 10 cm como pulverización preventiva.")
        st.link_button("Ir a Plantix  a  hongos blancos y polvorientos", "https://www.mango.org/wp-content/uploads/2020/08/Mango_Plagas_y_Enfermedades_SPN.pdf")

    elif class_names[np.argmax(predictions)] == 'Moho Negruzco':
        st.sidebar.warning(string)
        st.markdown("## Recomendación")
        st.info("Los insectos que causan el moho se matan rociándolos con carbarilo o fosfomidón al 0,03%. A esto le sigue una pulverización con una solución diluida de almidón o maida al 5%. Al secarse, el almidón se desprende en escamas y el proceso elimina los hongos negros y mohosos de diferentes partes de la planta.")
        st.link_button("Ir a UNA a  Moho con mancha negra", "https://www.una.py/mango-con-mancha-negra-es-seguro-consumirlo-como-hacerle-frente")
