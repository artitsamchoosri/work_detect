
import argparse
import io
import os
from PIL import Image
from IPython.display import display
import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, redirect, Response
import requests

app = Flask(__name__)

modelcheck= torch.hub.load('.', 'custom', path='best.pt', source='local',force_reload=True) 

# Set Model Settings

modelcheck.eval()
modelcheck.conf = 0.45  # confidence threshold (0-1)
modelcheck.iou = 0.45  # NMS IoU threshold (0-1) 
from io import BytesIO

def gen():
    
    cap=cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Read until video is completed
    count=0
    chap=0
    lines=0
    count_frames=0
    status=0
    persent=0
    product_ok=0
    product_ng=0
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        imgorg=frame
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            img = Image.open(io.BytesIO(frame))
            results = modelcheck(img, size=640)

            df=results.pandas().xyxy[0]
            df = df.sort_values(['ymin', 'xmin'],ascending=[False, True])
            (img_W, img_H) = (640, 640)
            (Y1, Y2) = (300, 340)

            #กรองเฉพาะจุดที่ตรวจได้ในระยะ
            df=df[(df['ymin']>Y1)]
            df=df[(df['ymin']<Y2)]
            #ตรวจสอบว่าจุดที่ตรวตสอบได้มี class fang อยู่หรือไม่
            #display(df)
            img = np.squeeze(results.render()) #RGB
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            
            if df.empty:
                print('DataFrame is empty!')
            else:
                df_sim=df.groupby(['name'])['name'].count().reset_index()
                display(df_sim)

            img_BGR = cv2.putText(img_BGR, 'Detected', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            imgs = cv2.line(img_BGR, (0,Y1), ((img_W-1),Y1), (255,255,0), 2)
            imgs = cv2.line(img_BGR, (0,Y2), ((img_W-1),Y2), (255,255,0), 2)

            #img_BGR = cv2.putText(img_BGR, 'Triangle = '+str(chap), (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            #img_BGR = cv2.putText(img_BGR, 'Circle = '+str(lines), (50,125), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            #img_BGR = cv2.putText(img_BGR, 'Square = '+str(persent), (50,150), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
        else:
            break

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

