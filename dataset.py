import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2


class GazeDataSet(Dataset):
    def __init__(self, filepath, split="train", transform=None, **args) -> None:
        self.transform = transform
        self.filepath = os.path.join(filepath, split)
        self.split = split

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
        image_path = self.list_files[index]
        # pilimg = Image.open(image_path).convert('L')
        # H,W = pilimg.width,pilimg.height
        if self.split != "test":
            label = self.list_labels[index]
            return image_path, ""
        else:
            return image_path, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    filepath = "D:\\dataset\\openEDS\\GazePrediction\\train\\sequences\\6400\\000.png"
    img = Image.open(filepath).convert('L')
    cv2.imshow("img", np.array(img))
    table = 255.0*(np.linspace(0, 1, 256)**0.8)
    img1 = cv2.LUT(np.array(img), table)
    cv2.imshow("img1", img1)
    cv2.waitKey(0)

    # ds = GazeDataSet("D:\\dataset\\openEDS\\GazePrediction",split="train")
    # for i in range(10):
    #     image_path,label = ds[i]
    #     img = Image.open(image_path).convert('L')
    #     cv2.imshow("img",np.array(img))
    #     cv2.waitKey(1000)
    #     # plt.imshow()
    #     # plt.show()
    #     print(label)
