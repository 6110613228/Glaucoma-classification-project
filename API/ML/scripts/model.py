import joblib
import cv2 as cv
import numpy as np
import tensorflow as tf
from .extract import features

def ml_predict(image):
    result = features(image) # [cdr, dcd, ex]
    model = load_model.model_ml

    result = np.array(result)
    result = result.reshape(1, -1) 

    prediction = model.predict_proba(result)

    predict_class = np.argmax(prediction)
    class_conf = prediction[0][predict_class]

    if predict_class == 0:
        class_out = 'glaucoma'
    elif predict_class == 1:
        class_out = 'normal'
    else:
        class_out = 'other'

    return [class_out, class_conf]
    
class load_model():
    model_ml = joblib.load('models/KNN.sav')
