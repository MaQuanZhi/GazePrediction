import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import feature_alpha_dropout
from torch.nn.modules.linear import Linear
# from resnet_m import resnet18, resnet50
from torchvision.models import resnet18


class GazePredictModel(nn.Module):
    def __init__(self):
        super(GazePredictModel,self).__init__()
        self.resnet = resnet18(pretrained=True,channels=1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,256)
        self.last_fc = nn.Linear(256,3)
        # self.fc = nn.Sequential(
        #     nn.Linear(1536,1024),
        #     nn.Linear(1024,256),
        # )
        # self.lstm = nn.LSTM(256, 256,bidirectional=True,num_layers=2,batch_first=True)
        # self.last_fc = nn.Linear(256,15)

    def forward(self,x):
        x = self.resnet(x)
        # x = self.fc(x)
        # x = self.lstm(x)
        x = self.last_fc(x)
        return x


if __name__ == "__main__": 
    # from tensorboardX import SummaryWriter
    model = GazePredictModel()
    model = model.cuda()
    # swriter = SummaryWriter(logdir="runs/model")
    # input_ = torch.zeros(1,1,400,640).cuda()
    # swriter.add_graph(model,input_to_model=input_,verbose=True)

    from torchsummary import summary  # summary无法正确识别lstm
    summary(model, input_size=(1,224,224))
    # nn.LSTM()

