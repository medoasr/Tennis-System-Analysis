import cv2

from utils import  (read_video,save_video)
from trackers import  PlayerTracker ,BallTracker
from Court_LIne_Detector import  CourtLineDetector
def main():
    # read Vid
    input_vidPath='sample/vidtest.mp4'
    vid_frames=read_video(input_vidPath)
    # Detect Player
    player_tracker=PlayerTracker('yolo11n.pt')
    player_detections=player_tracker.detect_frames(vid_frames,True,'tracker_stubs/player_detections.pkl')

    #detect Ball
    ball_tracker=BallTracker('models/best_ball_detection.pt')
    ball_detections=ball_tracker.detect_frames(vid_frames,True,'tracker_stubs/ball_detections.pkl')
    ball_detections=ball_tracker.interpolate(ball_detections)

    # detect Court_Kps
    court_detector=CourtLineDetector('models/keypoints_model.pth')
    keypoints=court_detector.predict(vid_frames[0])

    #choose Players
    player_detections=player_tracker.filter_players(keypoints,player_detections)
    # Draw Boxes
    new_frames=player_tracker.draw_boxes(vid_frames, player_detections)
    new_frames=ball_tracker.draw_boxes(new_frames, ball_detections)
    new_frames=court_detector.draw_keypoints_video(new_frames,keypoints)
    # Draw Frame Number
    for i ,frame in enumerate(new_frames):
        cv2.putText(frame,f"frame: {i}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    save_video(new_frames,'output_videos/Video.avi')


## 2:14
if __name__ =='__main__':
    main()