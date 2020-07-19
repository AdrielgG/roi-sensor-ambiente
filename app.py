import cv2
from bounding_box import bounding_box as bb
from datetime import datetime as dt
from farmertech.research.AI import CNN_Model
import numpy as np

cap = cv2.VideoCapture("../onepig1.mp4")

beberoudo = (554, 101, 166, 367)
cocho = (0, 0, 355, 212)

np.set_printoptions(suppress=True)

model = CNN_Model(keras_model='keras_model.h5',roi=cocho)

def main():
   while(True):
        ret, frame = cap.read()
        result = model.result(frame)
        data_value = result[0]
        image = cv2.resize(frame,(int(720),int(480)))
        bb.add(image, beberoudo[0], beberoudo[1], beberoudo[0]+beberoudo[2], beberoudo[1]+beberoudo[3], "bebedouro", "aqua")
        bb.add(image, cocho[0], cocho[1], cocho[0]+ cocho[2], cocho[1]+cocho[3], "COCHO STATUS: " + data_value, "orange")
        img = image
        #roi_cocho = result[1]
        #cv2.imshow("roi sensor",roi_cocho)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
