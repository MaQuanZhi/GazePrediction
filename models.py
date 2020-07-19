import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.nn.init import normal, constant
import math
from torchvision.models import resnet18




class GazeLSTM(nn.Module):
    def __init__(self):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc = nn.Linear(512, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)

        self.last_layer = nn.Linear(2*self.img_feature_dim, 3)


    def forward(self, input):

        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        # base_out = self.fc(base_out)
        base_out = base_out.view(input.size(0),25,self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,20:,:]
        output = self.last_layer(lstm_out)

        return output


if __name__ == "__main__":
    # model = resnet18()
    # device = torch.device('cuda')
    # model = model.to(device)
    # # print(model)
    # from torchsummary import summary
    # summary(model, input_size=(21,224, 224))
    from tensorboardX import SummaryWriter
    model = GazeLSTM()
    # model = model
    swriter = SummaryWriter(logdir="runs/model3")
    input_ = torch.zeros(2,75,224,224)
    swriter.add_graph(model,input_to_model=input_)