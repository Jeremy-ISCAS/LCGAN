import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import argparse
import os
import time
from logging import getLogger
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Net(nn.Module):
    def __init__(
        self,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        normalize=False,
        output_dim=128,
        hidden_mlp=2048,
        nmb_prototypes=0,
        eval_mode=False,
    ):
        super(Net, self).__init__()

        
        #swav ori res arch

        # encoder

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        # self.fc = nn.Linear(4096, 2048, bias=True)
        self.fc = nn.Linear(4096, 2048, bias=True)
        # self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

        
        #### ori res 
        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(2048, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(2048, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

                # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward_backbone(self, x):
        layer = 100
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        #x = x.view(x.shape[0], -1)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        out = self.fc(x)
        return out


    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        # print('nay')
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        
        return x
    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        # print(idx_crops)
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        return output, self.forward_head(output)

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out



def fc(**kwargs):
    return Net(**kwargs)

if __name__=="__main__":
    model=Net()
    model.cuda()
    print(model)
#     # train_dataset = datasets.ImageFolder(os.path.join('/data/cifar10-imageformat', "train"),transform=torchvision.transforms.ToTensor())
#     # train_loader = torch.utils.data.DataLoader(
#     #     train_dataset,batch_size=32
#     # )
#     # for iter_epoch, (inp, target) in enumerate(train_loader):
#         # output=model(inp)
#         # print(output.size())
#         # print(embedding.size())
    data=torch.randn(192,3,96,96)
    output=model(data)
    print(output.size())