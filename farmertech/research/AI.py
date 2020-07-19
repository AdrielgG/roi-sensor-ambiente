
import cv2
import numpy as np
from datetime import datetime as dt
import tensorflow.keras
from PIL import Image, ImageOps



class CNN_Model(object):
    def __init__(self,keras_model,roi):
        self.keras_model = keras_model # keras_model.h5
        self.model = tensorflow.keras.models.load_model(self.keras_model)
        self.roi = roi

    def result(self,frame):
        image = cv2.resize(frame,(int(720),int(480)))
        cocho = self.roi
        imCrop = image[cocho[1]:cocho[1]+cocho[3], cocho[0]:cocho[0]+cocho[2]]
        img = cv2.resize(imCrop,(int(224),int(224)))
        normalized_image_array = (img.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = self.model.predict(data)
        result = prediction.tolist()[0]
        pesos = [round(result[0],2) ,round(result[1],2)]
        data_value = "[ porco " + str(pesos[0]) + "] - [ fundo: "+ str(pesos[1])+"]"
        return [data_value]
