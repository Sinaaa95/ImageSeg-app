import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

IMAGE_SIZE = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = np.expand_dims(x, axis=-1)
    x = x/255.0
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

predictor_model = load_model('models/imgsegv2.h5', compile=False)

def predictor(img_path):
    x = read_image(img_path)
    y_pred = predictor_model.predict(np.expand_dims(x, axis=0))[0] > 0.5
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3))

    all_images = [
        x,
        white_line,
        mask_parse(y_pred)
    ]
    image = np.concatenate(all_images, axis=1)
    cv2.imwrite('outputs/output_image.png', image*255)

    output_path = 'outputs/output_image.png'

    return output_path

    #fig = plt.figure(figsize=(12, 12))
    #a = fig.add_subplot(1, 1)
    #imgplot = plt.imshow(image)





