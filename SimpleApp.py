import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np

#Título
st.title("Clasificación de imágenes")

#cargar el modelo

def load_model():
    model=tf.keras.models.load_model('/content/drive/MyDrive/Classroom/Modelado de Sistemas 24-Invierno MCC/basic_model.h5')
    return model

with st.spinner("Cargando modelo...."):
    model=load_model()

#clases de CIFAR-10
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# pre-procesamiento de la imagen
def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(28,28))
    img=tf.expand_dims(img,axis=0)
    return img

# Leer imagen desde URL dada por el usuario
image_path=st.text_input("URL de la imagen a clasificar...","https://media.istockphoto.com/photos/passenger-airplane-flying-above-clouds-during-sunset-picture-id155439315?k=20&m=155439315&s=612x612&w=0&h=BvXCpRLaP5h1NnvyYI_2iRtSM0Xsz2jQhAmZ7nA7abA=")

# Tomar imágen y predecir
if image_path:
    try:
        content=requests.get(image_path).content
        st.write("Predicción...")
        with st.spinner("Clasificación..."):
            img_tensor=load_image(content)
            pred=model.predict(img_tensor)
            pred_class=classes[np.argmax(pred)]
            st.write("Predicted Class:",pred_class)
            st.image(content,use_column_width=True)
    except:
        st.write("URL Inválida")
