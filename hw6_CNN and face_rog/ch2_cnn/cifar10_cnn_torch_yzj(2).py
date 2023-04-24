# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import sys
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

def setup_seed(seed):
    torch.manual_seed(seed) #设置cpu随机种子
    torch.cuda.manual_seed_all(seed) #设计gpu随机种子
    torch.backends.cudnn.deterministic = True #设置CuDNN随机种子

setup_seed(1)

# 定义超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 20

# 定义数据预处理方式
# 普通的数据预处理方式
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness= 0.1, contrast= 0.1, saturation= 0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
# 数据增强的数据预处理方式
# transform = transforms.Compose(

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])


# 定义数据集 -下载到workspace/dataset/private
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)


# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#定义3X3的卷积层
def conv3x3(in_channels, out_channels, stride=1):

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
# Resnet残差块
    def __init__(self, in_channels, out_channels, stride=1):
        #expansion
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))#传入out而非x，一开始传的是x
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    '''
    ResNet模型
    '''
    def __init__(self, block, num_layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.build_layer(block, 64, 64, num_layers[0], stride=1)
        self.layer2 = self.build_layer(block, 64, 128, num_layers[1], stride=2)
        self.layer3 = self.build_layer(block, 128, 256, num_layers[2], stride=2)
        self.layer4 = self.build_layer(block, 256, 512, num_layers[3], stride=2)
        self.mlp = nn.Linear(512, num_classes)

        #expansion~ planes并非输出维度

    def build_layer(self, block, in_channels, out_channels, num_layers, stride):
        strides = [stride] + [1]*(num_layers - 1)
        layers = []
        inc = in_channels
        outc = out_channels
        for stride in strides:
            layers.append(block(inc, outc, stride))
            inc = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(3,stride=2)(out)
        ###然后改resnet里的这个自适应池化为stride2,因为没有设置maxpool所以通过这一层池化来缩尺寸
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out

# 实例化模型
def ResNet18_cifar():
    return ResNet(ResidualBlock,[2,2,2,2])

use_mlu = True
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

model = ResNet18_cifar().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9,weight_decay=5e-4)
 

# 训练模型
for epoch in range(num_epochs):
    # 训练模式

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))