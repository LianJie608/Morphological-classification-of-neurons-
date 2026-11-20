import os
from shutil import copy
import random


random.seed(0)
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = './swc_data/3D_Img_human'
flower_class = [cla for cla in os.listdir(file_path)]
print(flower_class)

# 创建 训练集train 文件夹，并由类名在其目录下创建2个子目录
mkfile('./swc_data/3D_Img_human/train/')
for cla in flower_class:
    mkfile('./swc_data/3D_Img_human/train/' + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile('./swc_data/3D_Img_human/test/')
for cla in flower_class:
    mkfile('./swc_data/3D_Img_human/test/' + cla)

# 划分比例，训练集 : 测试集 = 7: 3
split_rate = 0.3

# 遍历所有类别的全部图像并按比例分成训练集和测试集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    print(num)
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存测试集test的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = './swc_data/3D_Img_human/test/' + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = './swc_data/3D_Img_human/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")




