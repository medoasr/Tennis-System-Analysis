import cv2

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret,frame=cap.read() #ret=false if no frames remains
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames,path):
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out=cv2.VideoWriter(path,fourcc,30,(output_frames[0].shape[1],output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()