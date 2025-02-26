import cv2
from ultralytics import YOLO
import pickle
import  pandas as pd
class BallTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
    def interpolate(self,bounding_boxes):
        ball_positions=[x.get(1,[]) for x in bounding_boxes]
        df=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df=df.interpolate()
        df=df.bfill()

        ball_positions=[{1:x} for x in df.to_numpy().tolist() ]
        return  ball_positions

    def detect_frames(self,frames,read_from_stub=False,stub_path=None):
        ball_detections=[]
        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb')as f:
                player_detections=pickle.load(f)
            return player_detections
        for frame in frames:
            player_dict=self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(ball_detections,f)
        return  ball_detections
    def detect_frame(self,frame):
        results=self.model.predict(frame,conf=0.2)[0]

        ball_dict={}

        for box in results.boxes:
            result=box.xyxy.tolist()[0]
            ball_dict[1]=result
        return  ball_dict



    def draw_boxes(self,vid_frames,ball_dict):
            new_frames=[]
            for frame,ball in zip(vid_frames,ball_dict):
                for track_id,box in ball.items():
                    x1,y1,x2,y2=box
                    cv2.putText(frame,f'Ball_id:{track_id}',(int(box[0]),int(box[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0))
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                new_frames.append(frame)
            return  new_frames

