#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import json
#import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.models
import tensorflow.keras as keras


# In[56]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.xception import Xception


# In[ ]:


st.write("""
# Image Classifier Prototype
""")

st.write("You can use our prepared images OR upload your own image below for test functionality")


# In[ ]:


@st.cache(allow_output_mutation=True)
def load_config(config_path: str):

    with open(config_path, 'r') as fr:
        config = json.load(fr)

    return config

@st.cache(allow_output_mutation=True)
def load_nn():
    # load weights
    #model.load_weights(model/best.hdf5)
    #model = keras.models.load_model('model/best')
    file = open('model/best', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    # load weights
    model.load_weights('model/best.hdf5')
    #model = load_model(weight_path)

    return model

#model_weight_path =  keras.models.load_model('model/best.h5')

class_indices  ={0: 'Acne and Rosacea Photos', 1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 2: 'Atopic Dermatitis Photos', 3: 'Bullous Disease Photos', 4: 'Cellulitis Impetigo and other Bacterial Infections', 5: 'Eczema Photos', 6: 'Exanthems and Drug Eruptions', 7: 'Hair Loss Photos Alopecia and other Hair Diseases', 8: 'Herpes HPV and other STDs Photos', 9: 'Light Diseases and Disorders of Pigmentation', 10: 'Lupus and other Connective Tissue diseases', 11: 'Melanoma Skin Cancer Nevi and Moles', 12: 'Nail Fungus and other Nail Disease', 13: 'Poison Ivy Photos and other Contact Dermatitis', 14: 'Psoriasis pictures Lichen Planus and related diseases', 15: 'Scabies Lyme Disease and other Infestations and Bites', 16: 'Seborrheic Keratoses and other Benign Tumors', 17: 'Systemic Disease', 18: 'Tinea Ringworm Candidiasis and other Fungal Infections', 19: 'Urticaria Hives', 20: 'Vascular Tumors', 21: 'Vasculitis Photos', 22: 'Warts Molluscum and other Viral Infections'}

# In[ ]:


def file_selector(folder_path='./images'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select Image:', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()

def preprocessing_image(image_pil_array: 'PIL.Image'):

    image_pil_array = image_pil_array.convert('RGB')
    image_pil_array = image_pil_array.resize((299,299))
    x = image.img_to_array(image_pil_array)

    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return test_datagen.flow(x)

if filename:
    img = Image.open(filename)
    st.image(img, caption="Your Image", use_column_width=True)
    model = load_nn()
    #model = keras.models.load_model('model/best.hdf5')
    x = preprocessing_image(img)
    label2 = model.predict(x)
    label = (np.asarray([label2]))[0]
    predict_rank =  (-label).argsort()[:3]
    st.write('The image is', (class_indices[predict_rank[0][0]]))
    st.write('Also it may be', (class_indices[predict_rank[0][1]]))
    st.write('Also it may be', (class_indices[predict_rank[0][2]]))
 #   st.write('The image is %s with %.5f%% probability' % (class_indices[predict_rank[0][0]], (predict_rank[2]) ))

st.write('Class Probability')

