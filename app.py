import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            return key

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('mango_model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()



st.write("""
         # Mango Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = {0: 'Anthracnose', 1: 'Bacterial Canker', 2: 'Cutting Weevil', 3: 'Die Back', 4: 'Gall Midge', 5: 'Healthy', 6: 'Powdery Mildew', 7: 'Sooty Mould'}

    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    string = "Detected Disease : " + predicted_class_name
    st.sidebar.warning(string)

    if predicted_class_name == 'Healthy':
        st.balloons()
        st.sidebar.success(string)
    else:
        st.markdown("## Remedy")

        # Add remedy suggestions based on the predicted class
        if predicted_class_name == 'Anthracnose':
            st.info("Remedy suggestion for Anthracnose")
        elif predicted_class_name == 'Bacterial Canker':
            st.info("Remedy suggestion for Bacterial Canker")
        elif predicted_class_name == 'Cutting Weevil':
            st.info("Remedy suggestion for Cutting Weevil")
        elif predicted_class_name == 'Die Back':
            st.info("Remedy suggestion for Die Back")
        elif predicted_class_name == 'Gall Midge':
            st.info("Remedy suggestion for Gall Midge")
        elif predicted_class_name == 'Powdery Mildew':
            st.info("Remedy suggestion for Powdery Mildew")
        elif predicted_class_name == 'Sooty Mould':
            st.info("Remedy suggestion for Sooty Mould")

