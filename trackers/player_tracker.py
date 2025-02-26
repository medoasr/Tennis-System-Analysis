import cv2
from ultralytics import YOLO
import pickle
import  sys
sys.path.append('../')
from utils import (distance,center_box)
class PlayerTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
    def detect_frames(self,frames,read_from_stub=False,stub_path=None):
        player_detections=[]
        if read_from_stub and stub_path is not None:
            with open(stub_path,'rb')as f:
                player_detections=pickle.load(f)
            return player_detections
        for frame in frames:
            player_dict=self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(player_detections,f)
        return  player_detections
    def detect_frame(self,frame):
        results=self.model.track(frame)[0]
        id_classes_dict=results.names

        player_dict={}

        for box in results.boxes:
            track_id=int(box.id.tolist()[0]) # tensor-> list->item
            result=box.xyxy.tolist()[0]
            obj_cls_id=box.cls.tolist()[0]
            obj_cls_name=id_classes_dict[obj_cls_id]
            if obj_cls_name=='person':
                player_dict[track_id]=result
        return  player_dict



    def draw_boxes(self,vid_frames,player_detections):
            new_frames=[]
            for frame,player_detection in zip(vid_frames,player_detections):
                for track_id,box in player_detection.items():
                    x1,y1,x2,y2=box
                    cv2.putText(frame,f'Player_ID:{track_id}',(int(box[0]),int(box[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0))
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                new_frames.append(frame)
            return  new_frames
    def filter_players(self,court_keypoints,player_detections):
        filtered_players=[]
        player_detections_first_frame=player_detections[0]
        chosen_players=self.choose_p(court_keypoints,player_detections_first_frame)
        for player_dict in player_detections:
            filtered_p={track_id:box  for track_id,box in player_dict.items() if track_id in chosen_players}
            filtered_players.append(filtered_p)
        return  filtered_players

    def choose_p(self,key_point,detect_firstframe):
        distances=[] # of each player id and court
        for id,box in detect_firstframe.items():
            player_center=center_box(box)
            min_distance=float('inf')
            for i in range(0,len(key_point),2):
                court_p=(key_point[i],key_point[i+1])
                distance_=distance(player_center,court_p)
                if distance_<min_distance:
                    min_distance=distance_
            distances.append((id,min_distance))
        distances.sort(key=lambda x:x[1])
        players=[distances[0][0],distances[1][0]]
        return players


