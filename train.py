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
import argparse
from losses import AngleLoss
from tensorboardX import SummaryWriter
import time
from torchvision import transforms as T
from itertools import chain

batch_size = 4
best_error = 100 # init with a large value
epochs = 100
count_test = 0
count = 0
workers = 0
checkpoint_test = 'checkpoint_GazeLSTM.pth.tar'
test_run = False



sWriter = SummaryWriter(logdir="runs/GazeLSTM")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        # print(path)
        return Image.new("RGB", (400, 640), "white")


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_list = AverageMeter()
    # prediction_error = AverageMeter()
    # angular = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i,  (source_frame,target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda()
        target = target.cuda()

        source_frame_var = torch.autograd.Variable(source_frame)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(source_frame_var)


        # loss = Variable(criterion(output, target_var),requires_grad=True)
        loss = criterion(output, target_var)

        loss_list.update(loss.item(), source_frame.size(0))

        sWriter.add_scalar("loss", loss_list.val, count)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        count = count +1

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_list))

def train_LSTMCell(train_loader, model, criterion,optimizer, epoch):
    global count
    loss_list = AverageMeter()
    # switch to train mode
    model.train()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    for i,  (img_path_list,label_list) in enumerate(train_loader):
        # label_list = label_list.permute(1,0,2) # 4,100,3 -> 100,4,3
        hx = torch.zeros(batch_size,256).cuda()
        cx = torch.zeros(batch_size,256).cuda()
        loss = 0
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
            loss += criterion(output,label_list_tensor)
            loss_list.update(loss.item())

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

        sWriter.add_scalar("loss", loss_list.val, count)
        count = count +1
            
        print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader),loss=loss_list))


def validate_LSTMCell(val_loader, model, criterion):
    global count_test
    loss_list = AverageMeter()
    model.eval()

    count = 0
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    for i,  (img_path_list,label_list) in enumerate(val_loader):
        img_list_tensor = torch.FloatTensor(batch_size,3,224,224)
        # label_list = label_list.permute(1,0,2) # 4,100,3 -> 100,4,3
        label_list_tensor = torch.FloatTensor(batch_size,6,3).cuda()
        hx = torch.zeros(batch_size,256).cuda()
        cx = torch.zeros(batch_size,256).cuda()
        loss = 0
        for j in range(len(img_path_list)-5):
            img_list = [transform(default_loader(img_path)) for img_path in img_path_list[j]]
            for k in range(batch_size):
                img_list_tensor[k,...] = img_list[k]
                label_list_tensor[k,...] = label_list[k,j:j+6,:]
            img_list_tensor = img_list_tensor.cuda()
            target_list_tensor = label_list_tensor.view(-1,18)
            output,hx,cx = model(img_list_tensor,hx,cx)
            loss += criterion(output,target_list_tensor)
            loss_list.update(loss.item())

            sWriter.add_scalar("loss", loss_list.val, count)
            count = count +1

        print('Epoch: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), loss=loss_list))
                   
    return loss_list.avg



def validate(val_loader, model, criterion):
    global count_test
    batch_time = AverageMeter()
    loss_list = AverageMeter()
    # prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    # angular = AverageMeter()

    for i, (source_frame,target) in enumerate(val_loader):

        source_frame = source_frame.cuda()
        target = target.cuda()

        source_frame_var = torch.autograd.Variable(source_frame,volatile=True)
        target_var = torch.autograd.Variable(target,volatile=True)
        with torch.no_grad():
            # compute output
            output = model(source_frame_var)

            loss = criterion(output, target_var)
            # angular_error = compute_angular_error(output,target_var)
            # pred_error = ang_error[:,0]*180/math.pi
            # pred_error = torch.mean(pred_error,0)

            # angular.update(angular_error, source_frame.size(0))
            # prediction_error.update(pred_error, source_frame.size(0))

            loss_list.update(loss.item(), source_frame.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


        print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                   loss=loss_list))

    # sWriter.add_scalar("predicted error", prediction_error.avg, count)
    # sWriter.add_scalar("angular-test", angular.avg, count)
    sWriter.add_scalar("loss-test", loss_list.avg, count)
    return loss_list.avg

def main():
    global args, best_error

    model = GazeLSTMCell().cuda()
    # model = torch.nn.DataParallel(model_v).cuda()
    # model.cuda()
    data_path = "C:\\mqz\\openEDS\\GazePrediction"
    torch.backends.cudnn.benchmark = True

    train_set = GazeDataSetPath(filepath=data_path,split="train")
    validation_set = GazeDataSetPath(filepath=data_path,split="validation")
    # train_loader = DataLoader(train_set,1)
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers)
    
    val_loader = DataLoader(
        validation_set, batch_size, shuffle=True, num_workers=workers)

    # criterion = AngleLoss().cuda()
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters())

    # checkpoint = torch.load(checkpoint_test)
    # model.load_state_dict(checkpoint['state_dict'])
    # epoch_start = checkpoint["epoch"] if checkpoint["epoch"] else 0
    epoch_start = 0
    for epoch in range(epoch_start, epochs):
        # train for one epoch
        train_LSTMCell(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        error = validate_LSTMCell(val_loader, model, criterion)

        # remember best angular error in validation and save checkpoint
        is_best = error < best_error
        best_error = min(error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_error,
        }, is_best,filename=f"checkpoint_GazeLSTMCell_{epoch+1}.pth.tar")

def save_checkpoint(state, is_best, filename='checkpoint_GazeLSTMCell.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_GazeLSTMCell.pth.tar')

if __name__ == "__main__":
    main()

