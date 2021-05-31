# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:04:43 2021

@author: 86198
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:50:07 2021

@author: 86198
"""

import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_data(path):
    #synthetic_dataset_path = r"spinal_diseases_diagnose/train/data"
    list = []
    for ind, file in enumerate(glob.glob(path + '/*.txt')):
        list.append(file)

    inputs_end = []
    location_end = []
    inputs = []
    location = []
    targets = []
    targets_end = []
    m = 0
    for i, file in enumerate(list):
        # print(file)
        m = m + 1
        input = []
        located = []
        target = []
        bbs = open(list[i]).readlines()
        # print(bbs)
        for j in range(len(bbs)):
            # print(j)
            labels = [0, 0, 0, 0, 0]
            a = bbs[j].split(',')
            x, y = int(a[0]), int(a[1])
            labels[0] = x
            labels[1] = y
            # p=bbs[j].find("disc':'v")
            p = bbs[j].find("disc': 'v")
            # print(p)
            if p == -1:  # 没有‘disc’,为第一类
                c = 0
                if len(a) == 4:
                    r = a[3].split(':')
                    s = r[1]
                    d = int(s[3])
                    e = 0
                    if s[4] != "'":
                        print("ERROR")

            else:
                c = 1
                # print(a[2])
                t = a[2].split(':')
                if len(t[1]) == 5:
                    u = t[1]
                    d = int(u[3])
                    e = 0
                else:
                    u = t[1]
                    d = int(u[3])
                    o = a[3]
                    e = int(o[1])
            labels[2] = c
            labels[3] = d
            labels[4] = e
            locate = [labels[0], labels[1]]
            located.append(locate)
            input.append(labels)
            if labels[2] == 0:
                target.append(labels[3] + 4)
                #targets_end.append(labels[3] + 4)
            elif labels[4] == 0:
                target.append(labels[3] - 1)
                #targets_end.append(labels[3] - 1)
            else:
                target.append(labels[3] + labels[4])
                #targets_end.append(labels[3] + labels[4])
        #inputs.append(input)
        #location.append(located)
        #targets.append(target)
        inputs_end.append(input)
        location_end.append(located)
        targets_end.append(target)          

    for batch_num in location_end:
            if len(batch_num) == 10:
                batch_num.append(batch_num[9])
                
    for batch_num in targets_end:
            if len(batch_num) == 10:
                batch_num.append(batch_num[9])
    
    labels=torch.tensor(targets_end)
    x,y=labels.shape
    labels=labels.reshape(x*y,1)
    labels=labels.squeeze()
    return location_end, labels
   
def get_testdata(path): 
    #path = r"spinal_diseases_diagnose/test/data"
    list = []
    for ind, file in enumerate(glob.glob(path + '/*.txt')):
        list.append(file)
    
    inputs_end = []
    location_end = []
    inputs = []
    location = []
    targets = []
    targets_end = []
    m = 0
    for i, file in enumerate(list):
            # print(file)
        m = m + 1
        input = []
        located = []
        target = []
        bbs = open(list[i]).readlines()
            # print(bbs)
        for j in range(len(bbs)):
                # print(j)
            labels = [0, 0, 0, 0, 0]
            a = bbs[j].split(',')
            x, y = int(a[0]), int(a[1])
            labels[0] = x
            labels[1] = y
                # p=bbs[j].find("disc':'v")
            p = bbs[j].find("disc': 'v")
            if p == -1:  # 没有‘disc’,为第一类 
                c=0
                s=a[3].split(':')[-1]
                d=int(s[3])
            else:
                c=1
                for w in a:
                    z=w.find("disc': 'v")
                    if  z!=-1:
                       t=w.split(':')[-1]
                       #print(t)
                       d=int(t[3])
                       #print(d)
            labels[2] = c
            labels[3] = d
            labels[4] = 0
            locate = [labels[0], labels[1]]
            located.append(locate)
            input.append(labels)
            if labels[2] == 0:
                target.append(labels[3] + 4)
                    #targets_end.append(labels[3] + 4)
            elif labels[4] == 0:
                target.append(labels[3] - 1)
                    #targets_end.append(labels[3] - 1)
            else:
                target.append(labels[3] + labels[4])
        
        location_end.append(located)
        targets_end.append(target)          
        for batch_num in location_end:
                if len(batch_num) == 10:
                    batch_num.append(batch_num[9])
                    
        for batch_num in targets_end:
                if len(batch_num) == 10:
                    batch_num.append(batch_num[9])
        
    labels=torch.tensor(targets_end)
    x,y=labels.shape
    labels=labels.reshape(x*y,1)
    labels=labels.squeeze()
    return location_end, labels

def get_img(pos,img_path):
     #img_path = r'spinal_diseases_diagnose\train\data\*jpg'
     imgs = glob.glob(img_path)
     img_all=[]
     for i,path in enumerate(imgs):
        img = Image.open(path)
        # 先统一图片大小
        img = img.resize((512, 512))
        img = np.array(img)
        for j,locate in enumerate(pos[i]):
            #print(j,locate)
            x,y=locate
            t=img[x-35:x+35,y-14:y+14]
            width,height=t.shape
            if height!=28:
                t=np.zeros((70,28))
            img_all.append(t)
            
     inputs=torch.tensor(img_all)
     inputs=inputs.unsqueeze(1)
     return inputs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)           #卷积层 输入通道为3，输出通道为6，卷积核大小为5*5，原来为1*70*28，卷积后为6*64*24
        self.pool = nn.MaxPool2d(2, 2)            #池化层，大小为2*2，第一次池化后为 6*32*12，第二次池化后为 16*14*4
        self.conv2 = nn.Conv2d(6, 16, 5)          #卷积层 输入通道为6，输出通道为16，卷积核大小为5*5，原来为6*32*12，卷积后为16*28*8
        self.fc1 = nn.Linear(16 * 14 * 4, 120)     #全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 14 * 4)                #view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()                                    #分类的交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)      # 随机梯度下降(使用momentum）


#主函数，网络训练过程
if __name__ == '__main__':
    synthetic_dataset_path = r"spinal_diseases_diagnose/train/data"
    train_img_path = r'spinal_diseases_diagnose\train\data\*jpg'
    test_img_path = r'spinal_diseases_diagnose\test\data\*jpg'
    pos,labels=get_data( synthetic_dataset_path )
    inputs=get_img(pos,train_img_path).float()
    for epoch in range(100):  # 神经网络参数更新次数

        running_loss = 0.0
        optimizer.zero_grad()                         # zero the parameter gradients

        outputs = net(inputs)                     #得到神经网络输出结果
        #print(outputs)
        loss = criterion(outputs, labels)         #计算损失结果
        loss.backward()                           #反向传播，计算loss的参数梯度
        optimizer.step()                          #更新神经网络的参数

        running_loss += loss.item()
        if epoch % 10 == 9:                  # print every 2000 mini-batches
            print('epoch: %d loss: %.3f' % (epoch + 1, running_loss ))
            running_loss = 0.0

    print('Finished Training')
    
    test_path = r"spinal_diseases_diagnose/test/data"
    
    pos_test,labels_test=get_testdata( test_path )
    inputs_test=get_img(pos,test_img_path).float()
    running_loss = 0.0
    optimizer.zero_grad()                         # zero the parameter gradients

    net.eval()
    outputs_test = net(inputs_test)                     #得到神经网络输出结果
    
    loss = criterion(outputs_test, labels_test)         #计算损失结果
    loss.backward()                           #反向传播，计算loss的参数梯度
    optimizer.step()                          #更新神经网络的参数
    running_loss += loss.item()
    print('test loss: %.3f' % running_loss)
    running_loss = 0.0
    
    correct=0
    total=0
    sum_loss=0
    _, predicted = torch.max(outputs_test.data, 1)
    for i,seq in enumerate(predicted):
        
        total=total+1
        sum_loss+=(predicted[i]-labels_test[i])**2
        if predicted[i]==labels_test[i]:
            correct=correct+1
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))