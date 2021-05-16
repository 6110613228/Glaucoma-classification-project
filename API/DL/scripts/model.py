import cv2 as cv
import numpy as np
import tensorflow as tf


def dl_predict(image):
    # load model
    model = load_model.model_dl

    # preprocess image (resize and expand dimention)
    image = cv.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Made prediction
    prediction = model.predict(image)

    # Get conf_soore and class out
    predict_class = np.argmax(prediction[0])
    class_conf = prediction[0][predict_class]

    if predict_class == 0:
        class_out = 'glaucoma'
    elif predict_class == 1:
        class_out = 'normal'
    else:
        class_out = 'other'

    return [class_out, class_conf]


class load_model():
    # Load DL_model here once
    model_dl = tf.keras.models.load_model('models/VGG16_35ep.h5')
