import random
import torch 
from logging import getLogger
from skimage import io,transform
import cv2
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

class MultiModalityDataset(Dataset):
    def __init__(
        self,root_dir,
        file_root_dir,
        transform=None,
        size_crops=[224, 96],
        nmb_crops=[2,6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14]):
        
        self.file_root_dir = file_root_dir   #文件目录
        self.transform = transform #变换
        self.root_dir=root_dir
        root=[]
        trans=[]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]         
        
        with open(file_root_dir) as f:
            for line in f.readlines():
                root.append(line)
        color_transform = [get_color_distortion(), RandomGaussianBlur()]
        
        for i in range(len(size_crops)):
            # print(i)
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        
        self.trans=trans
        self.images=root
    
    
    def __getitem__(self, index):
        image_index = self.images[index]#根据索引index获取该图片
        image_index1=image_index.split(' ')[0].strip()
        image_index2=image_index.split(' ')[1].strip()
        img_path1 = os.path.join(self.root_dir, image_index1)#获取索引为index的图片的路径名
        img_path2 = os.path.join(self.root_dir, image_index2)#获取索引为index的图片的路径名
        img1 = Image.open(img_path1)
        img2 =Image.open(img_path2)
        
      
        multi_crops=[]
        for i in range(len(self.trans)):
            # print('******')
            if i%2==0:
                tmp=self.trans[i](img1)
                multi_crops.append(tmp)
                
            else:
                tmp=self.trans[i](img2)
                multi_crops.append(tmp)
        
        return multi_crops
        
    def __len__(self):
        # 返回图像的数量
        # print("bbbb")
        return len(self.images)





class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort







if __name__ == "__main__":
    data_path='/data/multiview'
    ship_train_dataset = MultiModalityDataset(data_path,'/data/train3.txt')
    # print(type(ship_train_dataset[0]))
    # print(len(ship_train_dataset))
    dataloader=DataLoader(ship_train_dataset,batch_size=1,shuffle=None)


    for i_batch,batch_data in enumerate(dataloader):
        print(i_batch)#打印batch编号
        print(batch_data)
    