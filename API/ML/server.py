from fastapi import FastAPI, UploadFile, Form, File
import numpy as np
import cv2 as cv
from scripts.model import model

app = FastAPI()

@app.get('/')
def root():
    return {'msg' : 'Hello World'}

@app.post('/api/fundus')
async def upload_image(nonce: str=Form(None, title="Query Text"), image: UploadFile = File(...)):

    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    class_out, class_conf = model.ml_predict(img)

    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf)
    }
