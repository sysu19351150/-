''' 11目标检测网络训练文件 '''
import torch
import torchvision
import torchvision.transforms as transforms
import read_data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# 读入数据
path = r'spinal_diseases_diagnose\train\data'

# 运行read_data.load_data()函数获得符合输入fasterrcnn_resnet50_fpn网络的数据
# 其中，images是一个储存着以tensor形式储存图像数据的列表
# target是一个字典，包含”bbox“，”labels“项
# ”bbox“(``FloatTensor[N, 4]``)：对应的是储存有检测框左上、右下角点坐标的一个tensor，每行形如[x1, y1, x2, y2]
# ”labels“(``Int64Tensor[N]``): 对应的是每个box的label
images, targets = read_data.load_data(path)

# 定义模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=11)


#模型训练

# 由于小组计算机内存有限，无法完整跑完数据集，只对前60张图片进行了训练，得出效果较差
# 提交的代码中，解除了对输入图片数量的限制, 所得结果
images = images[:60]
targets = targets[:60]

output = model(images, targets)

#保存训练好的网络
torch.save(model, 'model_detect.pkl')


