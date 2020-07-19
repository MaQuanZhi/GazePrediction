import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from models import GazePredictModel
from dataset import GazeDataSet
from torch.utils.data import DataLoader
import argparse


def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.bs = 32
    args.workers = 1
    args.epochs = 10

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = False
    model = GazePredictModel()
    # print(model)
    model = model.to(device)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict, "models/GazeNet.pkl")
    model.train()
    nparams = get_nparams(model)
    from torchsummary import summary
    summary(model, input_size=(1, 400, 640))

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    train = GazeDataSet(split="train")
    # valid = GazeDataSet(split="validation")
    # test = GazeDataSet(split="test")
    trainloader = DataLoader(
        train, args.bs, shuffle=True, num_workers=args.workers)
    # validloader = DataLoader(valid,args.bs,shuffle=True,num_workers=args.workers)
    # testloader = DataLoader(valid,args.bs,shuffle=True,num_workers=args.workers)

    for epoch in range(args.epochs):
        for i, batchdata in enumerate(trainloader):
            image_path, label = batchdata
            # print(image_path,label)
            img = [np.expand_dims(np.array(Image.open(path).convert(
                'L'), dtype=np.float32), 0) for path in image_path]
            label = [np.array(_.strip().split(',')[1:], dtype=np.float32)
                     for _ in label]
            data = Variable(torch.tensor(img)).to(device)
            # print(data.shape)  # torch.Size([4, 1, 400, 640]) batch_size,通道数,高度,宽度
            # data = data.to(device)
            target = Variable(torch.tensor(label)).to(device)
            # data = img.to(device)
            # target = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            if i % 1 == 0:
                print('\rEpoch:{} [{}/{}], Loss: {:.9f}'.format(epoch,
                                                                i, len(trainloader), loss), end='')
            with open(f'loss.txt', 'a') as f:
                f.write("{},{},{}\n".format(epoch, i, loss))
            # if i%100 == 0:
            #     print('\rEpoch:{} [{}/{}], Loss: {:.3f}'.format(epoch,i,len(trainloader),loss),end='')
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"model_{epoch}_{loss:.9f}.pkl")
        '''
        保存checkpoint
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    ...
                    }, PATH)
        加载checkpoint
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # or
        model.train()
        '''

