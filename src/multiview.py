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
from skimage import color
class MultiviewDataset(Dataset):
    def __init__(
        self,root_dir,
        file_root_dir,
        transform=None,
        size_crops=[96, 96],
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
            #####sly 增添了一个PIL(mode='LAB')
            trans.extend([transforms.Compose([
                transforms.Resize(224),
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        
        self.trans=trans
        self.images=root
        # self.translab=transforms.Compose([transforms.ToPILImage(mode='HSV'),
        #                                   transforms.CenterCrop(224),
        #                                   transforms.ToTensor()])
        
        self.translab=transforms.ToPILImage(mode='LAB')
        self.transycbcr=transforms.ToPILImage(mode='YCbCr')
        
        self.ABtrans=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()])
    
    
    def __getitem__(self, index):
        image_index = self.images[index]#根据索引index获取该图片
        image_index=image_index.split(' ')[0].strip()
        img_path = os.path.join(self.root_dir, image_index)#获取索引为index的图片的路径名
        img = Image.open(img_path)

        multi_crops=[]

        returnimageA=RGB2YCbCr(img)  #return RGBgray
        # returnimageB=img
        returnimageA = self.transycbcr(returnimageA) 
        # returnimageA: PIL YCbCr view
        # img: PIL original view
        # img_B = self.transhsv(returnimageB)

        img_A = self.ABtrans(returnimageA)
        img_B = self.ABtrans(img)

        multi_crops.append(img_A)
        multi_crops.append(img_B)

        for i in range(2, len(self.trans)):
            # print('******')
            if i%2==0:
                tmp=self.trans[i](returnimageA)
                # tmp=self.trans[i](img)
                
                multi_crops.append(tmp)
                
            else:
                # tmp=self.trans[i](img)
                tmp=self.trans[i](img)
                # tmp=self.trans[i](img)
                multi_crops.append(tmp)
        
        # [A, B, a1, b1, a2, b2, a3, b3]
        
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


def RGB2HSV(img):
    if type(img) is torch.Tensor:
        is_tensor = True
        img_np = img.numpy()
    else:
        is_tensor = False
        img_np = img
    img_np = color.rgb2hsv(img_np)
    if is_tensor:
        img = torch.from_numpy(img_np)
    else:
        img = img_np
    return img

def RGB2Lab(img):
    if type(img) is torch.Tensor:
        is_tensor = True
        img_np = img.numpy()
    else:
        is_tensor = False
        img_np = img
    img_np = color.rgb2lab(img_np)
    if is_tensor:
        img = torch.from_numpy(img_np)
    else:
        img = img_np
    return img

def RGB2YCbCr(img):
    if type(img) is torch.Tensor:
        is_tensor = True
        img_np = img.numpy()
    else:
        is_tensor = False
        img_np = img
    img_np = color.rgb2ycbcr(img_np)
    if is_tensor:
        img = torch.from_numpy(img_np)
    else:
        img = img_np
    return img

if __name__ == "__main__":
    data_path='/data/multiview'
    ship_train_dataset = MultiModalityDataset(data_path,'/data/train3.txt')
    # print(type(ship_train_dataset[0]))
    # print(len(ship_train_dataset))
    dataloader=DataLoader(ship_train_dataset,batch_size=1,shuffle=None)


    for i_batch,batch_data in enumerate(dataloader):
        print(i_batch)#打印batch编号
        print(batch_data)
    