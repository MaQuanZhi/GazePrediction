import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision.models import resnet18




class GazeLSTM(nn.Module):
    def __init__(self):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc = nn.Linear(512, self.img_feature_dim)

        self.lstm_cell = nn.LSTMCell(self.img_feature_dim,self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
        
        self.fc = nn.Linear(self.img_feature_dim,3)

        self.last_layer = nn.Linear(2*self.img_feature_dim, 3)


    def forward(self, input):

        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        # base_out = self.fc(base_out)
        base_out = base_out.view(input.size(0),25,self.img_feature_dim)
        '''
        尝试使用LSTMCell model4
        base_out = base_out.permute(1,0,2) # 交换0，1轴,batch_size放中间
        hx = torch.randn(base_out.size()[1],base_out.size()[2])
        cx = torch.randn(base_out.size()[1],base_out.size()[2])
        output = []
        for i in range(base_out.size()[0]):
            hx,cx = self.lstm_cell(base_out[i],(hx,cx))
            out = self.fc(hx)
            output.append(out)
        '''
        lstm_out, _ = self.lstm(base_out)
        # lstm_out = lstm_out[:,20:,:]
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
    swriter = SummaryWriter(logdir="runs/model5")
    input_ = torch.zeros(2,75,224,224)
    swriter.add_graph(model,input_to_model=input_)