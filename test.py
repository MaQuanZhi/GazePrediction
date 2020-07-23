import os
import shutil
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from models import GazeLSTM
from dataset import GazeDataSet
from torch.utils.data import DataLoader
import argparse
from losses import AngleLoss
from tensorboardX import SummaryWriter
import time


def test(test_loader, model, criterion):
    model.eval()

    for i, (source_frame,label) in enumerate(test_loader):

        source_frame = source_frame.cuda()
        # target = target.cuda()
        # label = label.

        # source_frame_var = torch.autograd.Variable(source_frame,volatile=True)
        # target_var = torch.autograd.Variable(target,volatile=True)
        with torch.no_grad():
            # compute output
            output = model(source_frame)
            output = output[:,-5:,:]
            output_list = output.cpu().numpy().tolist()
            label_list = label.numpy().tolist()
            reslut_list = []
            # result_dict = {}
            for label_0,output_0 in zip(label_list,output_list):

            #     dict0 = {}
                for label_1,output_1 in zip(label_0,output_0):
                    # list0 = label_1+output_1
                    str0 = "%0.4d,%2d,%s,%s,%s"%(*label_1,*output_1)
                    reslut_list.append(str0)
            #         dict1 = {str(int(label_1[1])):output_1}
            #         dict0.update(dict1)
            #     result_dict.update({"%0.4d"%(int(label_1[0])):dict0})
            with open("result_test.txt",'a') as f:
                f.write("\n".join(reslut_list))
                f.write('\n')

            print("\n".join(reslut_list))
            # print(f"label:{label}\nprediction:{output}")
            # loss = criterion(output, target_var)
            # angleloss = AngleLoss()(output,target)
            # print(f"loss:{loss}\nangleloss:{angleloss}")

def main():
    model = GazeLSTM().cuda()

    data_path = "D:\\dataset\\openEDS\\GazePrediction"
    checkpoint_test = "model_best_GazeLSTM.pth.tar"
    batch_size = 8
    workers = 1
    torch.backends.cudnn.benchmark = True

    test_set = GazeDataSet(filepath=data_path,split="test",frame_num=20)
    test_loader = DataLoader(
            test_set, batch_size, shuffle=False, num_workers=workers)

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(checkpoint_test)
    model.load_state_dict(checkpoint['state_dict'])
    test(test_loader,model,criterion)

if __name__ == "__main__":
    main()

        