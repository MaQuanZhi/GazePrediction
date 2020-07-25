import os
import shutil
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from models import GazeLSTM, GazeLSTMCell
from dataset import GazeDataSet, GazeDataSetPath
from torch.utils.data import DataLoader
from losses import AngleLoss
from torchvision import transforms as T


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        print(path)
        # return Image.new("RGB", (400, 640), "white")

def test(test_loader, model):
    model.eval()

    for i, (source_frame,label) in enumerate(test_loader):

        source_frame = source_frame.cuda()

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


def test_LSTMCell(test_loader, model, batch_size):
    model.eval()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    for i,  (img_path_list,label_list) in enumerate(test_loader):
        # label_list = label_list.permute(1,0,2) # 4,100,3 -> 100,4,3
        subject = img_path_list[0][0].split('\\')[-2]
        hx = torch.zeros(batch_size,256).cuda()
        cx = torch.zeros(batch_size,256).cuda()
        # loss = 0
        with torch.no_grad():
            for j in range(len(img_path_list)):
                img_list_tensor = torch.FloatTensor(batch_size,3,224,224)
                img_list = [transform(default_loader(img_path)) for img_path in img_path_list[j]]
                # label_list_tensor = torch.FloatTensor(batch_size,6,3).cuda()
                for k in range(batch_size):
                    img_list_tensor[k,...] = img_list[k]
                    # label_list_tensor[k,...] = label_list[k,j:j+6,:]
                img_list_tensor = img_list_tensor.cuda()
                # label_list_tensor = label_list_tensor.view(-1,18)
                output,hx,cx = model(img_list_tensor,hx,cx)
                # loss += criterion(output,label_list_tensor)
                # loss_list.update(loss.item())
            out = output.view(6,3)[1:,...]
            out = out.cpu().numpy().tolist()
            result = ["%s,%2d,%s,%s,%s"%(subject,_+50,*out[_]) for _ in range(5)]
            print(result)
            # cpu().numpy()[0].tolist()
            # result = [f"{subject},{}"]
            # print(output)
            with open("result_val.txt",'a') as f:
                f.write("\n".join(result))
                f.write('\n')

        # print("\n".join(reslut_list))

def val_LSTMCell(test_loader, model, batch_size):
    model.eval()
    error_list = []
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    for i,  (img_path_list,label_list) in enumerate(test_loader):
        # label_list = label_list.permute(1,0,2) # 4,100,3 -> 100,4,3
        subject = img_path_list[0][0].split('\\')[-2]
        hx = torch.zeros(batch_size,256).cuda()
        cx = torch.zeros(batch_size,256).cuda()
        # loss = 0
        with torch.no_grad():
            for j in range(len(img_path_list)-5):
                img_list_tensor = torch.FloatTensor(batch_size,3,224,224)
                img_list = [transform(default_loader(img_path)) for img_path in img_path_list[j]]
                label_list_tensor = torch.FloatTensor(batch_size,6,3).cuda()
                for k in range(batch_size):
                    img_list_tensor[k,...] = img_list[k]
                    label_list_tensor[k,...] = label_list[k,j:j+6,:]
                img_list_tensor = img_list_tensor.cuda()
                label_list_tensor = label_list_tensor.view(-1,18)
                output,hx,cx = model(img_list_tensor,hx,cx)
                
               
            out = output.view(6,3)[1:,...]
            target = label_list_tensor.view(6,3)[1:,...]
            error = AngleLoss()(target,out)
            error_list.append(float(error))
            # print(error)
            print(error,torch.mean(torch.FloatTensor(error_list)))


def main_LSTMCell():
    model = GazeLSTMCell().cuda()

    data_path = "D:\\dataset\\openEDS\\GazePrediction"
    checkpoint_test = "model_best_GazeLSTMCell.pth.tar"
    batch_size = 1
    workers = 0
    torch.backends.cudnn.benchmark = True

    # test_set = GazeDataSetPath(filepath=data_path,split="test")
    # test_loader = DataLoader(
    #         test_set, batch_size, shuffle=False, num_workers=workers)
    val_set = GazeDataSetPath(filepath=data_path,split="validation")
    val_loader = DataLoader(
            val_set, batch_size, shuffle=False, num_workers=workers)

    checkpoint = torch.load(checkpoint_test)
    model.load_state_dict(checkpoint['state_dict'])
    val_LSTMCell(val_loader,model,batch_size)

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

    checkpoint = torch.load(checkpoint_test)
    model.load_state_dict(checkpoint['state_dict'])
    test(test_loader,model)

if __name__ == "__main__":
    # main()
    main_LSTMCell()

        