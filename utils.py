import cv2
import numpy as np
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
@register_keras_serializable()
def jaccard_index(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

def preprocess_for_model(image, target_size):
    img = cv2.resize(image, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)
