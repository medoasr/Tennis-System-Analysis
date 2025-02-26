import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import  json
import cv2
import numpy as np
device=("cuda" if torch.cuda.is_available() else "cpu")

'''
note the run was on colab to make use of gpu ^_^
'''
## Create Dataset
class KeyPointsDataSet(Dataset):
    def __init__(self,img_dir,labels_file):
        self.img_dir=img_dir
        with open(labels_file,'r') as f:
            self.data=json.load(f)
        self.transforms=transforms.Compose([
            transforms.ToPILImage,
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item=self.data[idx] # get annotation of item
        img=cv2.imread(f'{self.img_dir}/{item[id]}.png')
        w,h=img.shape[:2] # will need it to calc the new position of keypoints after resizing
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=self.transforms(img)
        kps=item['kps']
        kps=np.array(kps).flatten().astype(np.float16)
        kps[::2]*=224/w # adjust x coordinates for kps
        kps[1::2]*=224/h # adjust x coordinates for kps

        return img,kps

train_ds=KeyPointsDataSet('','')

train_loader=DataLoader(train_ds,8,shuffle=True)

### Create model
model=torchvision.models.resnet50(pretrained=True)

model.fc=torch.nn.Linear(model.fc.in_features,14*2) #14 Keypoints each one has an x and y

model=model.to(device)


#######################
#Start training
criterion=torch.nn.MSELoss() # as it is regression Prob
optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
epochs=20

model.train()
for epoch in range(epochs):
    for idx ,img,kps in enumerate(train_loader):
        img=img.to(device)
        kps=kps.to(device)
        optimizer.zero_grad()
        pred=model(img)

        loss=criterion(pred,kps)

        loss.backward()
        optimizer.step()

        if idx %10 ==0:
            print(f"Current Epoch {idx+1} , Current Loss {loss.item()}")


torch.save(model.state_dict(),'keypoints_model.pth')
##########################
