from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import GazeDataSet
import  argparse
from models import GazePredictModel
from torch.autograd import Variable
import numpy as np
import cv2

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.bs = 1
args.workers = 0
args.epochs = 10

device = torch.device("cuda")
model = GazePredictModel()
model.load_state_dict(torch.load('model_5_0.000110849.pkl'))
model = model.to(device)
model.eval()
valid = GazeDataSet(split="validation")
validloader = DataLoader(valid,args.bs,shuffle=True,num_workers=args.workers)
for i,batchdata in enumerate(validloader):
    if i >10:
        break
    image_path, label = batchdata
            # print(image_path,label)
    img = [np.expand_dims(np.array(Image.open(path).convert(
        'L'), dtype=np.float32), 0) for path in image_path]
    showimg = np.array(Image.open(image_path[0]).convert('L'))
    cv2.imshow(f"{label[0]}",showimg)
    cv2.waitKey(0)
    label = [np.array(_.strip().split(',')[1:], dtype=np.float32)
                for _ in label]
    data = Variable(torch.tensor(img)).to(device)
    output = model(data)
    print(label,output)
# print(model)