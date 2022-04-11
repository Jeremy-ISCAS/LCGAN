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
from src.multicropdataset import MultiCropDataset
from skimage import color


class ClassifyDataset(Dataset):
    def __init__(
        self, data_dir, 
        transform=None,
        size_crops=[96, 96],
        nmb_crops=[2,6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14]):

        trans=[]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform
        # print(data_info)
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
        
        self.translab=transforms.ToPILImage(mode='LAB')
        self.transycbcr=transforms.ToPILImage(mode='YCbCr')
        
        self.ABtrans=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()])
    
    def __getitem__(self, index):
        
        multi_crops=[]
        path_img, label = self.data_info[index]

        
        img = Image.open(path_img) # 读取图像，返回Image 类型 0~255
        
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

        # c

        return multi_crops, label
 
    def __len__(self):
        return len(self.data_info)
 

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
 
                # 遍历图片

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = sub_dir
                    data_info.append((path_img, int(label)))
        return data_info


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)
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