import torch
import torchvision.transforms as transforms
import torchvision.models as models
import  cv2

class CourtLineDetector:
    def __init__(self,model_path):
        self.model=models.resnet50(pretrained=False)
        self.model.fc=torch.nn.Linear(self.model.fc.in_features,14*2)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def predict(self,image):
            image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=self.transforms(image_rgb).unsqueeze(0) # to add batch dimension
            with torch.no_grad():

                output=self.model(image)

            key_points=output.squeeze().cpu().numpy() # remove all dim with (1) val
            originalh,originalw=image_rgb.shape[:2]
            key_points[::2]*=originalw/244.0
            key_points[1::2]*=originalh/244.0

            return  key_points

    def draw_ketpoints(self,frame,keypoints):
        for i in range(0,len(keypoints),2):
            x=int(keypoints[i])
            y=int(keypoints[i+1])
            cv2.putText(frame,f'{i//2}',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            cv2.circle(frame,(x,y),5,(255,0,0),-1)
        return  frame

    def draw_keypoints_video(self,vid_frames,keypoints):
        output_frames=[]
        for frame in vid_frames:
            frame=self.draw_ketpoints(frame,keypoints)
            output_frames.append(frame)
        return output_frames






