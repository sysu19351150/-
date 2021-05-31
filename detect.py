''' 11目标检测网络检验文件 '''
import torch
import numpy as np
import read_data
import torchvision.transforms as transforms
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# 图像处理器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# label值对应的关节名称
position =['L1', 'L2', 'L3', 'L4', 'L5', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1', 'T12-L1']

# 测试数据集所在地址
path = r'spinal_diseases_diagnose\test\data'

# 定义数据集获取函数，使得获取的数据集符合网络输入要求
def load_data(path):
    # imgs为读入的图片
    # pos 为 150 * 11 * 4 的数组，
    # 150个图片，每个图片11个关节， 用x,y两个坐标表示位置
    # tars 为 150 * 11， 记录了11个关节的类型，
    # height, weight储存图片的原始长宽
    imgs, pos, tars = read_data.get_data(path)
    pos = torch.tensor(pos)
    #pos = pos.reshape(150,11,4)
    tars = torch.tensor(tars, dtype=torch.int64)

    # print(pos,tars)
    images = []  # 图片列表
    width = []  # 宽高列表
    height = []

    # 对图像数据进行预处理
    for img in imgs:
        w,h = img.size
        width.append(w)
        height.append(h)
        # img = Image.open(i)
        # # 先统一图片大小
        img = img.resize((512, 512))
        img = np.array(img)
        temp = np.zeros((3,512,512))
        for i in range(3):
            temp[i, :, :] = img
        img = torch.tensor(temp/255., dtype=torch.float32)
        # print(img)
        # 存储图片tensor到images列表
        images.append(img)

    # 将数据重构，使得其符合网络要求的target输入格式
    targets = []
    for i in range(len(images)):
        d = {}
        d["boxes"] = pos[i]
        d["labels"] = tars[i]
        targets.append(d)

    return images, targets, width, height, imgs

# 调用函数数据集获取函数，获取检验所属数据集
images, targets, width, height, raw_img = load_data(path)

# 导入训练好的模型，开始测试
model = torch.load('model_detect.pkl')
model.eval()
output = model(images)


'''对测试所得结果进行提取，并储存和在图像上画出相应的标注'''
# 从output中提取相应的数据
boxes = []
scores = []
labels = []
for i in range(len(output)):
    boxes.append(output[i]['boxes'].detach().numpy().tolist())
    scores.append(output[i]['scores'].detach().numpy().tolist())
    labels.append(output[i]['labels'].detach().numpy().tolist())

# 对测试图像的地址，结果储存地址进行定义
paths = glob.glob(path + '\*.jpg')
img_paths = []
txt_paths = []
for file in paths:
    img_paths.append("test_result\\"+file.split("\\")[-1])
    txt_paths.append("test_result\\"+file.split("\\")[-1].replace(".jpg", ".txt"))


# 获取结果中的关节坐标的标志，将结果存入txt文件中，同时，在图像上标注出关节所在的点
for i in range(len(output)):
    #获取关节坐标
    label = []
    score = []
    pos = []
    for idx, lab in enumerate(labels[i]):
        if lab not in label:
            label.append(lab)
            score.append(scores[i][idx])
            x1,y1,x2,y2 = boxes[i][idx][0],boxes[i][idx][2],boxes[i][idx][1],boxes[i][idx][3]
            x, y = (x1+x2)/2, (y1+y2)/2
            pos.append([int(x*height[i]/512),int(y*width[i]/512)])
        else:
            if scores[i][idx]>score[label.index(lab)]:
                x1, y1, x2, y2 = boxes[i][idx][0], boxes[i][idx][2], boxes[i][idx][1], boxes[i][idx][3]
                x, y = (x1 + x2) / 2, (y1+y2) / 2
                pos[label.index(lab)] = [int(x), int(y)]
            else:
                continue
    # 将测试得出的关节坐标存入txt文件中
    with open(txt_paths[i],"w") as f:
        for j in range(len(label)):
            f.write(str(pos[j][0])+","+str(pos[j][1])+","+position[label[j]]+"\n")

    # 画出关节点
    img = Image.open(paths[i])
    draw = ImageDraw.Draw(img)
    for j in range(len(label)):
        draw.rectangle([pos[j][0]-2, pos[j][1]-2, pos[j][0]+2, pos[j][1]+2],outline = (255))
        font = ImageFont.truetype("simhei.ttf", 15, encoding="utf-8")
        draw.text([pos[j][0], pos[j][1]-5], position[label[j]], (255), font=font)
    img.save(img_paths[i])
