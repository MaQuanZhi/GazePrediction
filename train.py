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

batch_size = 2
best_error = 100 # init with a large value
epochs = 2
count_test = 0
count = 0
workers = 1
checkpoint_test = ''
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
        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)


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

def validate(val_loader, model, criterion):
    global count_test
    batch_time = AverageMeter()
    loss_list = AverageMeter()
    # prediction_error = AverageMeter()
    model.eval()
    end = time.time()
    # angular = AverageMeter()

    for i, (source_frame,target) in enumerate(val_loader):

        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)

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

    model = GazeLSTM().cuda()
    # model = torch.nn.DataParallel(model_v).cuda()
    # model.cuda()


    torch.backends.cudnn.benchmark = True

    train_set = GazeDataSet(split="train",frame_num=20)
    validation_set = GazeDataSet(split="validation",frame_num=20)

    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers)
    
    val_loader = DataLoader(
        validation_set, batch_size, shuffle=True, num_workers=workers)

    # criterion = AngleLoss().cuda()
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters())

    if test_run:
        test_set = GazeDataSet(split="test",frame_num=20)
        test_loader = DataLoader(
        test_set, batch_size, shuffle=False, num_workers=workers)
        checkpoint = torch.load(checkpoint_test)
        model.load_state_dict(checkpoint['state_dict'])
        angular_error = validate(test_loader, model, criterion)
        print('Angular Error is',angular_error)


    for epoch in range(0, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        angular_error = validate(val_loader, model, criterion)

        # remember best angular error in validation and save checkpoint
        is_best = angular_error < best_error
        best_error = min(angular_error, best_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_error,
        }, is_best)

def save_checkpoint(state, is_best, filename='checkpoint_GazeLSTM.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_GazeLSTM.pth.tar')

if __name__ == "__main__":
    main()


