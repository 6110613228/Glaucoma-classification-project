import joblib
import cv2 as cv
import numpy as np
from .extract import features

def ml_predict(image):
    try:
        result = features(image) # [cdr, dcd, ex]
    except(Exception, cv.error) as e :
        result = [0, 0, 0]
    
    class_label = ['glaucoma', 'normal', 'other']

    model = load_model.model_ml

    result = np.array(result)
    result = result.reshape(1, -1) 

    prediction = model.predict_proba(result)

    predict_class = np.argmax(prediction)
    class_conf = prediction[0][predict_class]

    class_out = class_label[predict_class]

    return [class_out, class_conf]
    
class load_model():
    model_ml = joblib.load('models/KNN.sav')
