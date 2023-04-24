# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import time
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# 定义数据预处理方式
# 普通的数据预处理方式
# transform = transforms.Compose([
#     transforms.ToTensor(),])
# 数据增强的数据预处理方式
transform = transforms.Compose(
[
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
)
#果然是由于数据增强的原因，导致了这个原因

# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ResNet(nn.Module):
    def __init__(self,block,layers):
        super(ResNet,self).__init__()
        self.inplanes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)#负责处理输入
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.BatchNorm1d(512,1),#最后一个小细节，防止特征爆炸
            nn.Linear(512,10)#适用于CIFAR10分类
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
    def _make_layer(self,block,planes,blocks,stride=1)->nn.Sequential:
        downsample=None
        if stride!=1 or self.inplanes!=planes:#这里还要乘一个block.expansion，但是没有更改过
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,planes,kernel_size=1,stride=stride),
                nn.BatchNorm2d(planes)#补0操作
            )
        layers=[]
        layers.append(
            block(self.inplanes,planes,stride,downsample)
        )#添加初始层
        self.inplanes=planes
        for _ in range(1,blocks):
            layers.append(
                block(self.inplanes,planes)
            )
        return nn.Sequential(*layers)

    def forward(self,x):
        #输入的处理
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        #中间卷积层
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        #输出
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)#相当于展平了
        x=self.fc(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self,inplanes,planes,str=1,downsample=None,base_width=64):
        super(ResidualBlock,self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=3,stride=str,padding=1)#补一层padding才能保证数据对齐
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1)#这样才能补上数据的对应结构
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=str
    def forward(self,x):
        identity=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)


        if self.downsample is not None:
            identity=self.downsample(x)
        out +=identity
        out=self.relu(out)
        return out

# 实例化模型
model = ResNet(ResidualBlock,[3,4,6,3])


use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("cuda is available")
    else:
        device = torch.device('cpu')

model = model.to(device)
writer=SummaryWriter("./logs_cifar10")
start_time=time.time()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion=criterion.to(device)
optimizer=torch.optim.Adam(model.parameters())
total_train_step = 0
total_test_step = 0
# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    # total=0
    # accuracy=0

    for data in train_loader:#输出列表和对应索引值
        images,labels=data
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # writer.add_graph(model,images)#显示网络架构
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()
        total_train_step=total_train_step+1
        # 打印训练信息
        if (total_train_step ) % 100 == 0:
            end_time=time.time()
            print(end_time-start_time,end=" ")
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Accuracy:{:2f}%'
                    .format(epoch + 1, num_epochs, total_train_step, len(train_loader), loss.item(),accuracy.item()*100))
            writer.add_scalar("train_loss",loss.item(),total_train_step)



    # 测试模式
    model.eval()
    total_test_lost=0

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss=criterion(outputs,labels)
            total_test_lost=total_test_lost+loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #total_accruacy:correct
            total_test_step=total_test_step+1

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        writer.add_scalar("test_loss",total_test_lost,total_test_step)
        writer.add_scalar("test_accuracy",100 * correct / total,total_test_step)
torch.save(model,"model_{}.pth".format(epoch))
print("模型已保存")

writer.close()