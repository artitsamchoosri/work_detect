
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
def getvalue(dataf,name):
    rslt_df = dataf.loc[dataf['name'] == name]
    if rslt_df.empty:
        return 0
    else:
        return int(rslt_df.iloc[0]['qty'])
def gen():
    
    cap=cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Read until video is completed
    triangle_now=0
    circle_now=0
    rectangle_now=0

    triangle_bf=0
    circle_bf=0
    rectangle_bf=0

    triangle_count=0
    circle_count=0
    rectangle_count=0

    while(cap.isOpened()):
        
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
            (Y1, Y2) = (200, 240)

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
                df_sim=df.groupby(['name'])['name'].count().reset_index(name='qty')
                #display(df_sim)

            if df.empty:
                triangle_now=0
                circle_now=0
                rectangle_now=0
                triangle_bf=0
                circle_bf=0
                rectangle_bf=0
            else:
                rectangle_now=getvalue(df_sim,"rectangle")
                triangle_now=getvalue(df_sim,"triangle")
                circle_now=getvalue(df_sim,"circle")

                if rectangle_now>rectangle_bf:
                    rectangle_bf=rectangle_now
                else:
                    rectangle_count+=rectangle_bf-rectangle_now
                    rectangle_bf=rectangle_now
                
                if triangle_now>triangle_bf:
                    triangle_bf=triangle_now
                else:



            img_BGR = cv2.putText(img_BGR, 'Detected', (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            cv2.line(img_BGR, (0,Y1), ((img_W-1),Y1), (255,255,0), 2)
            cv2.line(img_BGR, (0,Y2), ((img_W-1),Y2), (255,255,0), 2)

            #img_BGR = cv2.putText(img_BGR, 'Triangle = '+str(chap), (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            #img_BGR = cv2.putText(img_BGR, 'Circle = '+str(lines), (50,125), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
            #img_BGR = cv2.putText(img_BGR, 'Rectangle = '+str(persent), (50,150), cv2.FONT_HERSHEY_COMPLEX, 0.4,(0,255,255),1) 
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

