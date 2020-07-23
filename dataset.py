import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor
from torchvision import transforms as T


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        # print(path)
        return Image.new("RGB", (400, 640), "white")


def make_dataset(filepath="C:\\mqz\\openEDS\\GazePrediction", split="train", frame_num=50):
    filepath = os.path.join(filepath, split)
    dataset = []
    file_path_list = os.listdir(os.path.join(filepath, "sequences"))
    for file_path in file_path_list:
        for file in os.listdir(os.path.join(filepath, "sequences", file_path)):
            if file.endswith(".png"):
                file_number = int(file.split('.')[0])
                list_sources = []
                list_label = []
                if split == "test":
                    if file_number == 49:
                        for j in range(-frame_num+1, 1):
                            name_frame = os.path.join(
                                filepath, "sequences", file_path, '%0.3d.png' % (file_number+j))
                            list_sources.append(name_frame)
                        for i in range(50, 55):
                            list_label.append(f"test,{file_path},{i}")
                        dataset.append((list_sources, list_label))
                else:
                    if file_number >= frame_num-1:
                        for j in range(-frame_num+1, 1):
                            name_frame = os.path.join(
                                filepath, "sequences", file_path, '%0.3d.png' % (file_number+j))
                            list_sources.append(name_frame)
                        with open(os.path.join(filepath, "labels", f"{file_path}.txt"), 'r') as f:
                            lines = f.read().split('\n')
                            for line in lines[file_number-frame_num+1:file_number+6]:
                                if line:
                                    list_label.append(f"{file_path}\\{line}")
                        if len(list_label) == frame_num+5:
                            dataset.append((list_sources, list_label))
    return dataset


def make_dataset_path(filepath, split="train"):
    filepath = os.path.join(filepath, split)
    dataset = []

    file_path_list = os.listdir(os.path.join(filepath, "sequences"))
    for file_path in file_path_list:
        img_path_list = []
        # label_list =[]
        for img_path in os.listdir(os.path.join(filepath, "sequences", file_path)):
            img_path_list.append(os.path.join(
                filepath, "sequences", file_path, img_path))
        with open(os.path.join(filepath, "labels", f"{file_path}.txt"), 'r') as f:
            label_list = f.read().split('\n')[:-1]
        assert len(img_path_list) == len(label_list)
        dataset.append((img_path_list, label_list))
        # print(img_path_list)
    return dataset

class GazeDataSetPath(Dataset):

    def __init__(self, filepath, split="train") -> None:
        self.dataset = make_dataset_path(filepath, split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> set:
        img_path_list,label_list = self.dataset[index]
        label_list = [list(map(float,_.split(',')[1:])) for _ in label_list]
        label_list = torch.FloatTensor(label_list)
        return img_path_list,label_list




class GazeDataSetSingle(Dataset):
    '''
    return img, gaze_vector

    '''

    def __init__(self, filepath, split="train", transform=None, loader=default_loader, **args) -> None:
        self.transform = transform
        self.filepath = os.path.join(filepath, split)
        self.split = split
        self.loader = loader

        list_file_all = []
        list_label_all = []
        file_path_list = os.listdir(os.path.join(self.filepath, "sequences"))
        for file_path in file_path_list:
            for file in os.listdir(os.path.join(self.filepath, "sequences", file_path)):
                if file.endswith(".png"):
                    list_file_all.append(os.path.join(
                        self.filepath, "sequences", file_path, file))
            with open(os.path.join(self.filepath, "labels", f"{file_path}.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    list_label_all.append(f"{file_path}\\{line}")

        self.list_files = list_file_all
        self.list_labels = list_label_all
        self.testrun = args.get("testrun")

    def __len__(self) -> int:
        if self.testrun:
            return 10
        return len(self.list_files)

    def __getitem__(self, index: int) -> set:
        '''
        return img, gaze_vector
        '''
        image_path = self.list_files[index]
        img = torch.FloatTensor(3, 224, 224)
        # for i,frame_path in enumerate(image_path):
        img = self.transform(self.loader(image_path))
        # img = self.loader(image_path)

        if self.split != "test":
            line = self.list_labels[index]
            label = [float(_) for _ in line.split(',')[1:]]
            gaze_vector = torch.FloatTensor(label)
            return img, gaze_vector
        else:
            return img, torch.zeros(3)


class GazeDataSet(Dataset):
    '''
    return img, gaze_vector

    '''

    def __init__(self, filepath, split="train", frame_num=50,
                 transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]), loader=default_loader, **args) -> None:
        self.dataset = make_dataset(filepath, split, frame_num)
        self.transform = transform
        self.loader = loader
        self.frame_num = frame_num

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> set:
        '''
        return img, gaze_vector
        '''
        images_path, gaze = self.dataset[index]

        source_video = torch.FloatTensor(self.frame_num+5, 3, 224, 224)
        for i, image_path in enumerate(images_path):
            source_video[i, ...] = self.transform(self.loader(image_path))
        for i in range(self.frame_num, self.frame_num+5):  # 后5帧为空白帧
            source_video[i, ...] = self.transform(self.loader(""))
        source_video = source_video.view((self.frame_num+5)*3, 224, 224)
        gaze = [[float(_) for _ in line.split(',')[1:]] for line in gaze]
        gaze_vector = torch.FloatTensor(gaze)
        return source_video, gaze_vector


if __name__ == "__main__":
    # transform = T.Compose([T.Resize((224, 224)),
    #                        T.ToTensor()])
    # "C:\\mqz\\openEDS_small\\GazePrediction"
    from torch.utils.data import DataLoader
    train_set = GazeDataSetPath("D:\\dataset\\openEDS\\GazePrediction",
                           split="train")
    batch_size = 4
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=2)
   
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    from models import GazeLSTMCell
    import torch.nn as nn

    model = GazeLSTMCell().cuda()
    model.train()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for i,  (img_path_list,label_list) in enumerate(train_loader):
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

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()

            
    # img_path_list, label_list = train_set[0]
    # target = label_list[0:6]
    # print(target)

    # print(source_video.size(), gaze_vector.size())
    # print(gaze_vector)
    # print(make_dataset(split="validation"))
    # import matplotlib.pyplot as plt

    # transform = T.Compose([ T.RandomResizedCrop(400,(0.8,1)),
    #                         T.Resize(224),
    #                         T.ToTensor()])
    # ds = GazeDataSetSingle("D:\\dataset\\openEDS\\GazePrediction",split="train",transform=transform)
    # for i in range(3):
    #     img,gaze_vector = ds[i]
    #     # plt.imshow(img)
    #     print(img.size())
    #     plt.imshow(T.ToPILImage()(img))
    #     plt.show()
    #     print(gaze_vector)

    # 查看数据分布
    # ds = GazeDataSet("D:\\dataset\\openEDS\\GazePrediction",split="train")
    # from mpl_toolkits import mplot3d
    # xdata,ydata,zdata = [],[],[]
    # print(len(ds))
    # for item in ds:
    #     img,label = item
    #     name,x,y,z = label.split(',')
    #     xdata.append(float(x))
    #     ydata.append(float(y))
    #     zdata.append(float(z))
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection="3d")
    # ax.scatter(xdata,ydata,zdata)
    # plt.show()
