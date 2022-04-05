"""
2022/3/11
1.此文件只进行了bn特征值的提取，没有进行实际剪枝。
2.使用的ResNet结构也很奇怪，不是官方的适用ImageNet的结构，也不是常用的适用于cifar的结构。
3.使用两种方法进行bn层特征值提取，分别为
  zzc的方法:
    对于整个模型添加forward_hook,在forward_hook函数中进行整个前向传播过程，并记录bn层的特征(原本方法是错的，修改了半天)；
  新的方法:
    对于每个bn模块注册一个forward_hook，不用模拟整个前向传播，可以自动记录bn层的值，比较简单
4.经过反复的调试，终于将第一种方法改对了。。
  对比了两种方法获得的每层bn层的特征值并进行了相减，结果表明两种方法获得bn值相同，且最后输出结果也相同
5.后面进一步采用对每一个bn层添加一个hook的方法，更简单快捷。

"""


import tqdm
import torch
from torch import nn
from copy import deepcopy
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import dataloader
from Channel_selection import channel_selection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activations1 = []
activations2 = []

output_test1 = []
output_test2 = []
output_test3 = []
output_test4 = []


# --------------------------------------- 数据加载
# dataSet = 'CIFAR100'
dataSet = 'CIFAR10'

if dataSet == 'CIFAR10':
    data_path = './data/'

    mean = [0.4940607, 0.4850613, 0.45037037]
    std = [0.20085774, 0.19870903, 0.20153421]

elif dataSet == 'CIFAR100':
    data_path = './data/'
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

if dataSet == 'CIFAR100':
    print(123)
    train_data_set = datasets.CIFAR100(data_path, transform=transform, download=False, train=True)
    test_data_set = datasets.CIFAR100(data_path, transform=transform, download=True, train=False)

elif dataSet == 'CIFAR10':
    print(456)
    train_data_set = datasets.CIFAR10(data_path, transform=transform, download=False, train=True)
    test_data_set = datasets.CIFAR10(data_path, transform=transform, download=False, train=False)

train_data_loder = dataloader.DataLoader(train_data_set, batch_size=256, shuffle=True)
test_data_loder = dataloader.DataLoader(test_data_set, batch_size=256, shuffle=True)
# ---------------------------------------

# model = models.resnext50_32x4d(pretrained=False)
# model = models.resnext50_32x4d(pretrained=False)
# print(model)
# exit()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # ---------------------------------------------
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        # ---------------------------------------------
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel * self.expansion, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # ---------------------------------------------
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差
        super(Bottleneck, self).__init__()
        # ---------------------------------------------
        self.bn1 = nn.BatchNorm2d(in_channel)
        # 剪枝辅助
        self.select = channel_selection(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)

        # ---------------------------------------------
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)

        # ---------------------------------------------
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)

        # ---------------------------------------------
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, classes_num):
        super(ResNet, self).__init__()

        self.in_channel = 64

        # self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)

        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.bn1 = nn.BatchNorm2d(512 * block.expansion)
        self.select = channel_selection(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.bn3 = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, classes_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))

        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.bn1(out)
        out1 = self.select(out)
        out1 = self.relu(out1)
        out = F.relu(out1)

        out = self.avgpool(out)
        # out = self.bn3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def activation_hook(self, model, input, output):

        # 无需计算的blocks
        pass_Blocks = ['layer1', 'layer1.0', 'layer1.1', 'layer1.2',
                       'layer2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
                       'layer3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
                       'layer4', 'layer4.0', 'layer4.1', 'layer4.2',
                       'layer1.0.downsample', 'layer2.0.downsample',
                       'layer3.0.downsample', 'layer4.0.downsample',
                       'layer1.0.downsample.0', 'layer2.0.downsample.0',
                       'layer3.0.downsample.0', 'layer4.0.downsample.0',
                       '']

        # 需要计算downsample的blocks
        RESIDUALS = ['layer1.0.downsample.0', 'layer2.0.downsample.0',
                     'layer3.0.downsample.0', 'layer4.0.downsample.0']

        # 需要记录residual的地方1
        IF_RESIDUAL1 = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0']

        # 需要记录residual的地方2
        IF_RESIDUAL2 = ['layer1.1', 'layer1.2',
                        'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
                        'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
                        'layer4.0', 'layer4.1', 'layer4.2',
                        'bn1']


        x = input[0]
        cov_residual_flag = False
        for name, module in model.named_modules():
            # 记录downsample层未改变前的x值
            if name in IF_RESIDUAL1:
                residual = x

            # 进行downsample层的计算
            if name in RESIDUALS:
                residual = module(residual)
                x += residual
                cov_residual_flag = False

            # 对于非downsample层，进行残差计算
            if name in IF_RESIDUAL2:

                if cov_residual_flag:
                    x += residual2

                residual2 = x
                cov_residual_flag = True

            # 跳过不用计算的模块
            if name in pass_Blocks:
                continue

            if isinstance(module, nn.Linear):
                x = x.view(x.size(0), -1)

            x = module(x)

            if isinstance(module, nn.BatchNorm2d):
                activations1.append(x.sum(dim=0))

            if isinstance(module, nn.Linear):
                output_test1.append(x)
                output_test4.append(output)

        return


# 读取每个类别图片pics_nun张
def read_Img_by_class(target_class, pics_num):

    data_loder = train_data_loder

    counts = []
    inputs = []

    for idx in range(len(target_class)):
        counts.append(0)

    # image_num表示微调时选区的图片数量
    for data, label in data_loder:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in tqdm.tqdm(range(len(label)), desc="loading_imgs: "):
            if label[idx] in target_class and counts[label[idx]] < pics_num:
                inputs.append(data[idx].to(device))
                counts[label[idx]] += 1

    imgs = torch.empty(len(inputs), 3, 32, 32).to(device)

    for idx, img in enumerate(inputs):
        imgs[idx] = img

    return imgs


def acvition_hook_test(model, input, output):
    activations2.append(output.clone().detach().sum(dim=0))
    return


def get_mask(imgs, model):

    list_activation1 = []
    list_activation2 = []

    imgs_masks = []
    thresholds = []
    one_img_mask = []
    hooks = []

    for img in imgs:
        thresholds.clear()
        activations1.clear()
        activations2.clear()
        one_img_mask.clear()
        hooks.clear()

        img = torch.unsqueeze(img, 0)

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(acvition_hook_test)
                hooks.append(hook)

        output = model(img)
        output_test2.append(output)
        for hook in hooks:
            hook.remove()

        hook_model = model.register_forward_hook(model.activation_hook)
        output = model(img)
        output_test3.append(output)
        hook_model.remove()

        list_activation1.append(activations1)
        list_activation2.append(activations2)

    return list_activation1, list_activation2


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], classes_num=num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], classes_num=num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], classes_num=num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], classes_num=num_classes)


model = resnet50(num_classes=10).to(device)

# num = 0
# for name, module in model.named_modules():
#     print(name)
#     # print(name, "------------", module)
#     if isinstance(module, nn.BatchNorm2d):
#         num += 1
# print(num)
# exit(0)


classes = [0]
imgs = read_Img_by_class(target_class=classes, pics_num=1)
# print(imgs.size())

ac1, ac2 = get_mask(imgs=imgs, model=model)

print(output_test1[0])
print(output_test2[0])
print(output_test3[0])
print(output_test4[0])

for i, j in zip(ac1[0], ac2[0]):
    print(torch.flatten(i-j).sum())
print(len(ac1[0]), len(ac2[0]))
# print(ac1[0][1], ac2[0][1])
