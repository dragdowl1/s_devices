
import cv2
import numpy as np



def process_image(frame, model):
    col, row = (640, 640)
    frame = frame.convert('RGB') 
    img = np.array(frame) 
    img = img[:, :, ::-1].copy() 

    out = model(img, size=640)
    jsn = [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],  
                "confidence": np.round(float(pred[4]),2),
            }
            for pred in result
        ]
        for result in out.xyxy
    ]
    return jsn
