# -- coding: utf-8 --
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
from image_processing import normalise_zero_one, reshape_zero_padding
import random
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 4 classes
# data_dir = {'train': ('./swc_data/3D_Img_raw/train/Glia','./swc_data/3D_Img_raw/train/interneuron',  './swc_data/3D_Img_raw/train/principal cell',  './swc_data/3D_Img_raw/train/sensory receptor'),
#             'test': ('./swc_data/3D_Img_raw/test/Glia','./swc_data/3D_Img_raw/test/interneuron',  './swc_data/3D_Img_raw/test/principal cell', './swc_data/3D_Img_raw/test/sensory receptor')}


#12分类
data_dir = {'train':
                (
                './swc_data/3D_Img_raw/train/interneuron/GABAergic',    './swc_data/3D_Img_raw/train/interneuron/Nitrergic',
                './swc_data/3D_Img_raw/train/principal cell/Parachromaffin',    './swc_data/3D_Img_raw/train/principal cell/Purkinje',
                './swc_data/3D_Img_raw/train/Glia/astrocyte',   './swc_data/3D_Img_raw/train/interneuron/basket',
                './swc_data/3D_Img_raw/train/principal cell/ganglion',  './swc_data/3D_Img_raw/train/principal cell/granule',
                './swc_data/3D_Img_raw/train/principal cell/medium spiny',  './swc_data/3D_Img_raw/train/Glia/microglia',
                './swc_data/3D_Img_raw/train/principal cell/pyramidal', './swc_data/3D_Img_raw/train/sensory receptor'),


            'test':
                (
                './swc_data/3D_Img_raw/test/interneuron/GABAergic', './swc_data/3D_Img_raw/test/interneuron/Nitrergic',
                './swc_data/3D_Img_raw/test/principal cell/Parachromaffin', './swc_data/3D_Img_raw/test/principal cell/Purkinje',
                './swc_data/3D_Img_raw/test/Glia/astrocyte', './swc_data/3D_Img_raw/test/interneuron/basket',
                './swc_data/3D_Img_raw/test/principal cell/ganglion',   './swc_data/3D_Img_raw/test/principal cell/granule',
                './swc_data/3D_Img_raw/test/principal cell/medium spiny',   './swc_data/3D_Img_raw/test/Glia/microglia',
                './swc_data/3D_Img_raw/test/principal cell/pyramidal',  './swc_data/3D_Img_raw/test/sensory receptor'),
            }

input_size =112
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(degrees=30),  # 随机旋转图像
        transforms.RandomHorizontalFlip(), # 随机水平翻转图像
        transforms.RandomVerticalFlip(),   # 随机垂直翻转图像
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
}


class neuronSet(Dataset):

    def __init__(self, phase, data_dir=data_dir):
        super(neuronSet, self).__init__()
        self.phase = phase
        self.data_dir = data_dir[phase]
        self.file_list = self.list_file()


    def list_file(self):
        # 返回所有文件路径的列表
        # 格式为[[第一类的所有文件路径]，[第二类的所有文件路径]， ...]
        # 并考虑了样本均衡
        file_list = [[] for d in self.data_dir]

        for i in range(len(file_list)):
            self.recur_listdir(self.data_dir[i], file_list[i])

        # # 样本均衡， 将所有类的样本数扩充到一致
        # if self.phase == 'train':
        #     m = max([len(fi) for fi in file_list])
        #     for fi in file_list:
        #         if len(fi) < m:
        #             extra = np.random.choice(fi, size=m - len(fi))
        #             fi.extend(extra)

        print('pngset ', self.phase, [len(fi) for fi in file_list])

        return file_list


    def __getitem__(self, item):
        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c + 1]):
            c += 1
        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]

        datum = Image.open(filename).convert('RGB')
        datum = data_transforms[self.phase](datum)

        # 将图像转换为numpy数组
        datum = np.array(datum)

        # 在深度维度上堆叠112次，即复制112份
        #将二维图像数据扩展为三维数据，将输入的二维图像沿着特定的深度维度进行复制，形成一个三维数据结构
        depth =112
        datum = np.stack([datum] * depth, axis=1)

        label = c
        return datum, label


    def __len__(self):
        return sum([len(l) for l in self.file_list])

    def recur_listdir(self, path, dir_list):
        # 将path下的所有文件放到dir_list 中
        for f in os.listdir(path):
            if os.path.isdir(path + '/' + f):
                self.recur_listdir(path + '/' + f, dir_list)
            else:
                dir_list.append(path + '/' + f)

    def getitem(self, item):
        '''
        similar to __gititem__, but return the neuro_id of sample
        used for multi-model
        :return: neuro_id
        '''
        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c + 1]):
            c += 1

        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]
        neuro_id = int(filename.split('/')[-1].split('.')[0])
        #print ('neuro_id_',neuro_id)
        return neuro_id


if __name__ == '__main__':
    import os
    import random

    c = neuronSet('train')

    # 获取第一个数据
    datum, label = c[1]

    # 输出数据的形状
    print("Datum shape:", datum.shape)
    print(datum)

    # 输出标签
    print("Label:", label)


