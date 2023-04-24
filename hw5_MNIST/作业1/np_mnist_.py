# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm


# 加载数据集,numpy格式
#这是多分类问题啊
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(x,0)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    x[x<=0]=0
    x[x>0]=1
    return x

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    #分类头，注意为多分类
    exps=np.exp(x)#为了防止出现nan打破上限
    return exps/np.sum(exps)

def f_prime(x):
    '''
    softmax函数的导数
    '''
    #但这是不准确的啊，有些还不会返回这个值啊
    return f(x)*(1-f(x))

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    delta=1e-7
    return -(np.sum(y_true*np.log(y_pred+delta)))#添加一个微小值防止除0

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred-y_true
    # return -y_true/y_pred


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape,)

def get_weights(shape=()):
    '''
    生成权重矩阵
    '''
    np.random.seed(seed=0)
    return np.random.normal(loc=0.0, scale=0.01, size=shape, )

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=10):
        '''
        初始化网络结构
        '''
        self.w1=get_weights((input_size,hidden_size))
        self.b1=get_weights((hidden_size))
        self.w2=get_weights((hidden_size,output_size))
        self.b2=get_weights((output_size))
        self.lr=lr

    def forward(self, x):
        '''
        前向传播
        '''
        z1=np.matmul(x,self.w1)+self.b1
        a1=relu(z1)
        z2=np.matmul(a1,self.w2)+self.b2
        a2=f(z2)#加一个softmax层
        return z1,a1,z2,a2
    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        batch_size=0
        batch_loss=0
        batch_acc=0
        self.grads_w2=np.zeros_like(self.w2)
        self.grads_b2=np.zeros_like(self.b2)
        self.grads_w1=np.zeros_like(self.w1)
        self.grads_b1=np.zeros_like(self.b1)
        for x,y in zip(x_batch,y_batch):
            z1,a1,z2,a2=self.forward(x)
            #a2就是预测的数据
            loss=loss_fn(y,a2)
            delta_L=loss_fn_prime(y,a2)*f_prime(z2)
            delta_l=np.matmul(self.w2,delta_L)*relu_prime(z1)

            self.grads_w2+=np.matmul(np.array([a1]).T,[delta_L])
            self.grads_b2+=delta_L
            self.grads_w1+=np.matmul(np.array([x]).T,[delta_l])
            self.grads_b1+=delta_l

            batch_size+=1
            batch_loss+=loss
            pred=np.argmax(a2)
            sign=np.argmax(y)
            batch_acc+=(pred==sign)
        self.grads_b1/=batch_size
        self.grads_b2/=batch_size
        self.grads_w1/=batch_size
        self.grads_w2/=batch_size
        batch_acc/=batch_size
        batch_loss/=batch_size
        # print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))
        # 反向传播
        self.w2-=self.lr*self.grads_w2
        self.b2-=self.lr*self.grads_b2
        self.w1-=self.lr*self.grads_w1
        self.b1-=self.lr*self.grads_b1
        return batch_loss,batch_acc
    def eval(self,x_batch,y_batch):
        batch_size = 0
        batch_loss = 0
        batch_acc = 0
        for x, y in zip(x_batch, y_batch):
            z1, a1, z2, a2 = self.forward(x)
            # a2就是预测的数据
            loss = loss_fn(y, a2)
            batch_size += 1
            batch_loss += loss
            pred = np.argmax(a2)
            sign = np.argmax(y)
            batch_acc += (pred == sign)
        batch_acc /= batch_size
        batch_loss /= batch_size
        # print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))
        return batch_loss, batch_acc
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
if __name__ == '__main__':

    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=1)
    for epoch in range(10):
        np.random.seed(seed=1)
        print("epoch:{} train----start\n".format(epoch))
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:#每次只传入一个向量，这应该是打包，这就相当于一个dataloader，一次加载64个batch
            x_batch=X_train[i:i+64]
            y_batch=y_train[i:i+64]
            loss,acc=net.step(x_batch,y_batch)
            losses.append(loss)
            accuracies.append(acc)
        train_loss=np.mean(losses)
        train_acc=np.mean(accuracies)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print("train_losses:{} train_accuracy{}\n".format(train_loss,train_acc))
        print("epoch:{} train----end\n".format(epoch))
        print("epoch:{} test----start\n".format(epoch))
        test_losses=[]
        test_accuracies=[]
        # p_bar = tqdm(range(0, len(X_test), 64))
        # for i in p_bar:  # 每次只传入一个向量，这应该是打包，这就相当于一个dataloader，一次加载64个batch
        for i in range(0,len(X_test),64):
            tx_batch=X_test[i:i+64]
            ty_batch=y_test[i:i+64]
            tloss,tacc=net.eval(tx_batch,ty_batch)
            test_losses.append(tloss)
            test_accuracies.append(tacc)
        _test_losses=np.mean(test_losses)
        _test_acc=np.mean(test_accuracies)
        history['val_loss'].append(_test_losses)
        history['val_acc'].append(100. * _test_acc)
        print("\ntest_losses:{} test_accuracy{}\n".format(_test_losses,_test_acc))
        print("epoch:{} test----end\n".format(epoch))
# 画图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()
plt.show()

