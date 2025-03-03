# Tennis-System-Analysis
 use machine learning, computer vision and deep learning to create a tennis analysis system. This project utilizes YOlO a state of the art object detector to detect the players and the tennis balls. It also utilizes trackers to track those object across frames. 
## link of [tutorial](https://www.youtube.com/watch?v=L23oIHZE14w)

## Outputs
Here is a screenshot from one of the output videos:

![Screenshot](output_videos/screenshot.jpeg)

## Models Used
* YOLO v11n for player detection
* Fine Tuned YOLO for tennis ball detection
* Court Key point extraction

* Trained YOLOV5 model: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
* Trained tennis court key point model: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing

## Training
* Tennis ball detetcor with YOLO: training/tennis_ball_detector_training.ipynb
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Requirements
* python3.8
* ultralytics
* pytroch
* pandas
* numpy 
* opencv
