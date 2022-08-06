"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
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


#'''
# Load Pre-trained Model
#model = torch.hub.load(
       # "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
      #  )#.autoshape()  # force_reload = recache latest code
#'''
# Load Custom Model
#model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.pt", force_reload=True).autoshape()
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',force_reload=True,device='cuda')
modelscan = torch.hub.load('.', 'custom', path='best640v5s.pt', source='local',force_reload=True) 
modelcheck= torch.hub.load('.', 'custom', path='fang3px640.pt', source='local',force_reload=True) 
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',force_reload=True,source='local')
# Set Model Settings
modelscan.eval()
modelscan.conf = 0.6  # confidence threshold (0-1)
modelscan.iou = 0.45  # NMS IoU threshold (0-1) 
modelcheck.eval()
modelcheck.conf = 0.45  # confidence threshold (0-1)
modelcheck.iou = 0.45  # NMS IoU threshold (0-1) 
from io import BytesIO

def gen():
    
    cap=cv2.VideoCapture(0,cv2.CAP_QT)
    #cap.set(cv2.CAP_PROP_FPS, 30.0)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #1080, 720, 360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #1920, 1280, 640
    cap.set(cv2.CAP_PROP_FPS, 50)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(28, 85) 
    cap.set(cv2.CAP_PROP_EXPOSURE,-6.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
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
            if status==1:
                results = modelcheck(img, size=640)
            else:
                results = modelscan(img, size=640)

            

            df=results.pandas().xyxy[0]
            df = df.sort_values(['ymin', 'xmin'],ascending=[False, True])
            (img_W, img_H) = (640, 640)
            (Y1, Y2) = (100, 400)

            #กรองเฉพาะจุดที่ตรวจได้ในระยะ
            df=df[(df['ymin']>Y1)]
            df=df[(df['ymin']<Y2)]
            #ตรวจสอบว่าจุดที่ตรวตสอบได้มี class fang อยู่หรือไม่
            
            if (df['name'] == 'fang').sum()==0:
                 img_BGR = imgorg
                 status=0
            else:
                img = np.squeeze(results.render()) #RGB
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
                status=1
            
            if (df['name'] == 'fang').sum()==0:
                img_BGR = cv2.putText(img_BGR, 'Not found in Range', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,255),1) 
            else:
                img_BGR = cv2.putText(img_BGR, 'Detected', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            imgs = cv2.line(img_BGR, (0,Y1), ((img_W-1),Y1), (255,255,0), 2)
            imgs = cv2.line(img_BGR, (0,Y2), ((img_W-1),Y2), (255,255,0), 2)

            
            if status==1:
                count_frames += 1
                if (df['name'] == 'chap').sum()>0:
                    chap += 1
                if (df['name'] == 'lines').sum()>0:
                    lines += 1 
                try:
                    persent=round(max(chap/count_frames,lines/count_frames),2)
                except ZeroDivisionError:
                    persent = 0 
            elif status==0:
                if persent>=0.2 and count_frames>0:
                    product_ng+=1
                    #response = requests.get('http://10.10.20.153/0/ture')
                    #print(response.headers)
                elif persent<0.2 and count_frames>0:
                    product_ok+=1
                    #response = requests.get('http://10.10.20.153/1/false')
                    #print(response.headers)
                count_frames=0
                persent=0
                lines=0
                chap=0

            img_BGR = cv2.putText(img_BGR, 'Count Frames = '+str(count_frames), (50,75), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            img_BGR = cv2.putText(img_BGR, 'Chap = '+str(chap), (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            img_BGR = cv2.putText(img_BGR, 'Lines = '+str(lines), (50,125), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            img_BGR = cv2.putText(img_BGR, 'Persent = '+str(persent), (50,150), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            img_BGR = cv2.putText(img_BGR, 'OK = '+str(product_ok), (400,75), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            img_BGR = cv2.putText(img_BGR, 'NG = '+str(product_ng), (400,100), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
        else:
            break
        #print(cv2.imencode('.jpg', img)[1])

        #print(b)
        #frame = img_byte_arr

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

