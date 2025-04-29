import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class Liyutong(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 去掉ResNet最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.final_conv = nn.Conv2d(2048, num_classes, kernel_size=1)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b,c,h,w = x.shape
        # 使用ResNet提取特征
        # dim = 2048
        x = self.resnet(x)
        # print(x.shape)
        # 使用1x1卷积生成显著图
        x = self.final_conv(x)
        x = F.interpolate(x, (h,w), mode='bilinear', align_corners=True)
        return x

if __name__ == "__main__":
    model = liyutong().cuda()
    x = torch.randn(1,3,384,384).cuda()
    print(model(x).shape)