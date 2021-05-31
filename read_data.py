''' 11目标检测数据读取文件 '''
import glob
import numpy as np
from PIL import Image
import torch

def read_data(path):
    # 根据给定的地址，读入图片和txt文件中所给的target
    list1 = []
    for ind, file in enumerate(glob.glob(path + '\*.txt')):
        list1.append(file)

    # 定义array，labels，images
    # array中储存关节的标记
    # labels中储存关节的坐标点
    array = [[] for i in range(len(list1))]
    labels = []
    images = []

    # 读取给定的jpg文件和txt文件中的数据
    for i, element in enumerate(list1):
        label=[]
        element = element.replace('.txt','.jpg')
        img = Image.open(element)
        images.append(img)
        w, h = img.size
        lines = open(list1[i]).readlines()
        for j in range(len(lines)):
            str1 = lines[j].split(",")

            # 后面对图像的处理中将图像大小重构，此处将关节坐标x,y相应重构
            x, y =int(int(str1[0])*512/w), int(int(str1[1])*512/h)
            str = lines[j].split("'")
            in_index = str.index('identification')
            array[i].append(str[in_index+2])

            # 框出关节对应的box
            label.append([x-30,y-15,x+30,y+15])
        labels.append(label)
    
    for batch_num in array:
        if len(batch_num) == 10:
            batch_num.append(batch_num[9])

    for batch_num in labels:
        if len(batch_num) == 10:
            batch_num.append(batch_num[9])

    Array = np.zeros([len(list1), 11])
    for i in range(len(array)):
         for j in range(len(array[i])):
             str = array[i][j]
             Array[i][j] = reflect(str)


    return Array, labels, images

def reflect(str):
    # 根据读入的字符串，给出相应的label
    if str == 'L1':
        return 0
    if str == 'L2':
        return 1
    if str == 'L3':
        return 2
    if str == 'L4':
        return 3
    if str == 'L5':
        return 4
    if str == 'L1-L2':
        return 5
    if str == 'L2-L3':
        return 6
    if str == 'L3-L4':
        return 7
    if str == 'L4-L5':
        return 8
    if str == 'L5-S1':
        return 9
    if str == 'T12-L1':
        return 10
    else:
        return 11


def get_data(path):
    #返回读取后经初步处理的数据集
    arr, pos, images = read_data(path)
    return images, pos, arr

def load_data(path):
    # 本函数返回符合fastrcnn网络读取要求的数据集

    # imgs为读入的图片
    # pos 为 150 * 11 * 4 的数组，
    # 150个图片，每个图片11个关节， 用x,y两个坐标表示位置
    # tars 为 150 * 11， 记录了11个关节的类型，
    imgs, pos, tars = get_data(path)
    pos = torch.tensor(pos)
    tars = torch.tensor(tars, dtype=torch.int64)
    images = []  # 图片列表

    for img in imgs:
        # img = Image.open(i)
        # # 先统一图片大小
        img = img.resize((512, 512))
        img = np.array(img)
        temp = np.zeros((3,512,512))
        for i in range(3):
            temp[i, :, :] = img
        img = torch.tensor(temp/255., dtype=torch.float32)

        # 存储图片tensor到images列表
        images.append(img)

    targets = []
    for i in range(len(images)):
        d = {}
        d["boxes"] = pos[i]
        d["labels"] = tars[i]
        targets.append(d)

    return images, targets


