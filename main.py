from helper import *
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
sns.set_theme(style="darkgrid")
sns.set()
from PIL import Image
st.title('Image Segmentation Tool')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
    

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image
        #display_image = Image.open(uploaded_file)
        #display_image = cv2.imread(uploaded_file, cv2.IMREAD_UNCHANGED)
        #st.image(display_image)
        prediction = predictor(os.path.join('static/images',uploaded_file.name))
        os.remove('static/images/'+uploaded_file.name)
        # deleting uploaded saved picture after prediction
        # drawing graphs

        st.text('Predictions :-')

        #fig = plt.figure(figsize=(12, 12))
        #a = fig.add_subplot(1, 1, 1)
        #imgplot = plt.imshow(prediction)
        #st.imshow(fig)

        display_image = Image.open(prediction)
        #display_image = cv2.imread(uploaded_file, cv2.IMREAD_UNCHANGED)
        st.image(display_image)
