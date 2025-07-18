import tensorflow as tf
import cv2
import numpy as np

model=tf.keras.models.load_model("model/violence_detection_model.h5")

def preprocesss_frame(frame,image_size=(224,224)):
    resized=cv2.resize(frame,image_size)
    normalized=resized/255.0
    return np.expand_dims(normalized,axis=0)


def predict_frame(frame):
    preprocessed=preprocesss_frame(frame)
    prediction=model.predict(preprocessed)[0][0]
    return prediction
