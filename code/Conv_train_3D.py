import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
# from thop import profile
# from torchstat import stat
# from ptflops import get_model_complexity_info

from Img_input_3D import neuronSet
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from Conv_model_3D import cnn3d
# from Conv_model_3D import cnn3d_multiscale

##对比
from C3DNet import get_model
from Resnet_18 import generate_model
from DAM import Duo_Attention
from convnext import convnext_xtiny,ConvNeXt
from VIT import ViT3D

from Swin import SwinTransformerBlock3D
from Mednext import MedNeXt


# from ShuffleNet import get_model
# from mobilenet import get_model
from EfficientNet3D import EfficientNet3D



from torchsummary import summary
import os
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support, classification_report

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def get_accuracy(outputs, labels, batch_size):
    _, predicted = torch.max(outputs, 1)  # 返回每行中最大值及其索引
    correct = (predicted == labels).sum().item()
    accuracy = correct / batch_size
    return accuracy


# def get_metrics(outputs, labels):
#     pred = torch.argmax(outputs, dim=1).cpu()
#     labels = labels.cpu()
#
#     # 使用sklearn避免手动计算错误
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels, pred, average='macro', zero_division=0  # 处理除0错误
#     )
#     return precision.item(), recall.item(), f1.item()



def main():
    # 12分类  没有进行线性插值的
    # data_dir = {'train':
    #     ('./swc_data/3D_Img_raw_1/train/interneuron/GABAergic','./swc_data/3D_Img_raw_1/train/interneuron/Nitrergic',
    #     './swc_data/3D_Img_raw_1/train/principal cell/Parachromaffin','./swc_data/3D_Img_raw_1/train/principal cell/Purkinje',
    #     './swc_data/3D_Img_raw_1/train/Glia/astrocyte', './swc_data/3D_Img_raw_1/train/interneuron/basket',
    #     './swc_data/3D_Img_raw_1/train/principal cell/ganglion','./swc_data/3D_Img_raw_1/train/principal cell/granule',
    #     './swc_data/3D_Img_raw_1/train/principal cell/medium spiny','./swc_data/3D_Img_raw_1/train/Glia/microglia',
    #     './swc_data/3D_Img_raw_1/train/principal cell/pyramidal','./swc_data/3D_Img_raw_1/train/sensory receptor'),
    #     'test':
    #     ('./swc_data/3D_Img_raw_1/test/interneuron/GABAergic','./swc_data/3D_Img_raw_1/test/interneuron/Nitrergic',
    #     './swc_data/3D_Img_raw_1/test/principal cell/Parachromaffin','./swc_data/3D_Img_raw_1/test/principal cell/Purkinje',
    #     './swc_data/3D_Img_raw_1/test/Glia/astrocyte', './swc_data/3D_Img_raw_1/test/interneuron/basket',
    #     './swc_data/3D_Img_raw_1/test/principal cell/ganglion','./swc_data/3D_Img_raw_1/test/principal cell/granule',
    #     './swc_data/3D_Img_raw_1/test/principal cell/medium spiny','./swc_data/3D_Img_raw_1/test/Glia/microglia',
    #     './swc_data/3D_Img_raw_1/test/principal cell/pyramidal','./swc_data/3D_Img_raw_1/test/sensory receptor'),
    #      }

    # 4分类  没有进行线性插值的
    # data_dir = {'train': ('./swc_data/3D_Img_raw_1/train/Glia', './swc_data/3D_Img_raw_1/train/interneuron',
    #                       './swc_data/3D_Img_raw_1/train/principal cell', './swc_data/3D_Img_raw_1/train/sensory receptor'),
    #             'test': ('./swc_data/3D_Img_raw_1/test/Glia', './swc_data/3D_Img_raw_1/test/interneuron',
    #                      './swc_data/3D_Img_raw_1/test/principal cell', './swc_data/3D_Img_raw_1/test/sensory receptor')}

    # #swc_fixed数据集
    # data_dir = {'train':
    #     ('./swc_data/3D_Img_fixed_1/train/interneuron/GABAergic', './swc_data/3D_Img_fixed_1/train/interneuron/Nitrergic',
    #         './swc_data/3D_Img_fixed_1/train/principal cell/Parachromaffin','./swc_data/3D_Img_fixed_1/train/principal cell/Purkinje',
    #         './swc_data/3D_Img_fixed_1/train/Glia/astrocyte', './swc_data/3D_Img_fixed_1/train/interneuron/basket',
    #         './swc_data/3D_Img_fixed_1/train/principal cell/ganglion', './swc_data/3D_Img_fixed_1/train/principal cell/granule',
    #         './swc_data/3D_Img_fixed_1/train/principal cell/medium spiny', './swc_data/3D_Img_fixed_1/train/Glia/microglia',
    #         './swc_data/3D_Img_fixed_1/train/principal cell/pyramidal', './swc_data/3D_Img_fixed_1/train/sensory receptor'),
    #     'test':
    #         ('./swc_data/3D_Img_fixed_1/test/interneuron/GABAergic', './swc_data/3D_Img_fixed_1/test/interneuron/Nitrergic',
    #         './swc_data/3D_Img_fixed_1/test/principal cell/Parachromaffin','./swc_data/3D_Img_fixed_1/test/principal cell/Purkinje',
    #         './swc_data/3D_Img_fixed_1/test/Glia/astrocyte', './swc_data/3D_Img_fixed_1/test/interneuron/basket',
    #         './swc_data/3D_Img_fixed_1/test/principal cell/ganglion','./swc_data/3D_Img_fixed_1/test/principal cell/granule',
    #         './swc_data/3D_Img_fixed_1/test/principal cell/medium spiny', './swc_data/3D_Img_fixed_1/test/Glia/microglia',
    #         './swc_data/3D_Img_fixed_1/test/principal cell/pyramidal', './swc_data/3D_Img_fixed_1/test/sensory receptor'),
    # }

    # ##4分类
    # data_dir = {'train': ('./swc_data/3D_Img_fixed_1/train/Glia', './swc_data/3D_Img_fixed_1/train/interneuron',
    #                       './swc_data/3D_Img_fixed_1/train/principal cell',
    #                       './swc_data/3D_Img_fixed_1/train/sensory receptor'),
    #             'test': ('./swc_data/3D_Img_fixed_1/test/Glia', './swc_data/3D_Img_fixed_1/test/interneuron',
    #                      './swc_data/3D_Img_fixed_1/test/principal cell',
    #                      './swc_data/3D_Img_fixed_1/test/sensory receptor')}
    # ##human-6分类
    #
    data_dir = {'train': ( './swc_data/3D_Img_human_1/train/glutamatergic','./swc_data/3D_Img_human_1/train/granule',
                          './swc_data/3D_Img_human_1/train/induced pluripotent stem cell-(iPSC)-derived', './swc_data/3D_Img_human_1/train/pyramidal 7 Allen cell type',
                          './swc_data/3D_Img_human_1/train/sensory receptor',   './swc_data/3D_Img_human_1/train/von Economo',
                          ),

                'test': ( './swc_data/3D_Img_human_1/test/glutamatergic','./swc_data/3D_Img_human_1/test/granule',
                          './swc_data/3D_Img_human_1/test/induced pluripotent stem cell-(iPSC)-derived', './swc_data/3D_Img_human_1/test/pyramidal 7 Allen cell type',
                          './swc_data/3D_Img_human_1/test/sensory receptor',   './swc_data/3D_Img_human_1/test/von Economo',
                          ),
                }

    lr=1e-4
    # momentum=0.90
    epochs = 50
    batch_size = 16
    trained = False
    model_filename = 'model_2class_v1'
    num_classes = len(data_dir['train'])
    print(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainset = neuronSet('train', data_dir=data_dir)
    testset = neuronSet('test', data_dir=data_dir)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    if trained:
        model = torch.load(model_filename)
    else:
        # model = cnn3d()
        # model=cnn3d_multiscale()

        ##C3Dnet
        model = get_model(sample_size=112, sample_duration=16, num_classes=6, in_channels=3)

        # resnet-18
        # model =generate_model(model_depth=18,n_classes=6,n_input_channels=3, shortcut_type='B', conv1_t_size=7,
        #                              conv1_t_stride=1, no_max_pool=False,  widen_factor=1.0)

        # # Model configuration  3d-dam
        # model = Duo_Attention(num_classes=6,dropout=0)

        # ##convnext
        # model = convnext_xtiny(num_classes=12)


        # shufflenet
        # model = get_model(groups=3, num_classes=6, width_mult=1, in_channels=3)

        # mobilenet
        # model = get_model(sample_size=112, num_classes=12, in_channels=3)

        # efficientnet
        # model = EfficientNet3D.from_name('efficientnet-b4', override_params={'num_classes': 4}, in_channels=3)

        #VIT
        # model=ViT3D( image_size=(112, 112, 112),
        #     patch_size=16,
        #     num_classes=12,
        #     dim=1024,
        #     depth=6,
        #     heads=16,
        #     mlp_dim=2048,
        #     channels=3,
        #     dropout=0.1,
        #     emb_dropout=0.1
        #
        #     )
        # # # 实例化Swin3D模型
        # 比如你要做10分类
        # model = SwinTransformerBlock3D(
        #     dim=3,
        #     num_heads=3,
        #     input_resolution=(112, 112, 112),
        #     window_size=(2, 4, 4),
        #     num_classes=6,  # 在这里修改分类数量
        #     mlp_ratio=4.0,
        #     drop=0.1,
        #     attn_drop=0.1,
        #     drop_path=0.2
        # )

        # model = MedNeXt(in_chans=3, num_classes=6)


        model.to(device)
        print(model)

    #设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    summary(model, input_size=(3, 112, 112, 112))


    # macs, params = get_model_complexity_info(model, (3, 112, 112, 112), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    # 定义保存最高准确率对应的模型权重的变量
    # best_acc = 0.0
    #
    # # 定义保存训练集和测试集的loss和准确率的列表
    # train_loss_list = []
    # train_acc_list = []
    # train_precision_list = []
    # train_recall_list = []
    # train_f1_list = []
    #
    # test_loss_list = []
    # test_acc_list = []
    # # test_precision_list = []
    # # test_recall_list = []
    # # test_f1_list = []
    # train_start_time = time.time()
    #
    # # 训练模型
    # for epoch in range(epochs):
    #     epoch_start_time = time.time()
    #     for phase in ['train', 'test']:
    #         epoch_accuracy = 0
    #         epoch_loss = 0
    #         # epoch_precision = 0
    #         # epoch_recall = 0
    #         # epoch_f1 = 0
    #
    #         if phase == 'train':
    #             model.train()
    #             loader = trainloader
    #         else:
    #             model.eval()
    #             loader = testloader
    #         for i, batch in enumerate(loader):
    #             optimizer.zero_grad()
    #             data, labels = batch
    #             if data.shape[0] < batch_size:
    #                 continue
    #             if device:
    #                 data, labels = data.cuda(), labels.cuda()
    #                 # 调整维度顺序：从(B, C, T, H, W)转换为(B, T, H, W, C)
    #                 # data = data.permute(0, 2, 3, 4, 1).contiguous()
    #
    #             outputs = model(data)
    #             # outputs = model(data.view(8, 112 * 112 * 112, 3))
    #
    #             accuracy = get_accuracy(outputs, labels, batch_size)  # 准确率
    #             loss = criterion(outputs, labels.long())
    #             # precision, recall, f1 = get_metrics(outputs, labels)  # 精确率、召回率、F1值
    #
    #             epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
    #             epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)
    #             # epoch_precision = (epoch_precision * i + precision) / (i + 1)
    #             # epoch_recall = (epoch_recall * i + recall) / (i + 1)
    #             # epoch_f1 = (epoch_f1 * i + f1) / (i + 1)
    #
    #             if phase == 'train':
    #                 loss.backward()
    #                 optimizer.step()
    #
    #         # 保存训练集和测试集的loss和准确率
    #         if phase == 'train':
    #             train_loss_list.append(epoch_loss)
    #             train_acc_list.append(epoch_accuracy)
    #             # train_precision_list.append(epoch_precision)
    #             # train_recall_list.append(epoch_recall)
    #             # train_f1_list.append(epoch_f1)
    #         else:
    #             test_loss_list.append(epoch_loss)
    #             test_acc_list.append(epoch_accuracy)
    #             # test_precision_list.append(epoch_precision)
    #             # test_recall_list.append(epoch_recall)
    #             # test_f1_list.append(epoch_f1)
    #
    #         # 保存最高准确率对应的模型权重
    #         if phase == 'test' and epoch_accuracy > best_acc:
    #             best_acc = epoch_accuracy
    #             print(f'save best model,第{epoch + 1}轮')
    #             torch.save(model.state_dict(), './weight/best_model12_v1.pth')
    #     #
    #     # print('Epoch [{}/{}], Train Accuracy: {:.4f}, Train loss: {:.4f}, Train Precision: {:.4f}, Train Recall: {:.4f}, Train F1: {:.4f}'.format(
    #     #         epoch + 1, epochs, train_acc_list[-1], train_loss_list[-1], train_precision_list[-1],
    #     #         train_recall_list[-1], train_f1_list[-1]))
    #     # print('Epoch [{}/{}], Test Accuracy: {:.4f}, Test loss: {:.4f}, Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1: {:.4f}'.format(
    #     #         epoch + 1, epochs, test_acc_list[-1], test_loss_list[-1], test_precision_list[-1], test_recall_list[-1],
    #     #         test_f1_list[-1]))
    #
    #
    #     print('Epoch [{}/{}], Train Accuracy: {:.4f}, Train loss: {:.4f}'.format(epoch + 1, epochs, train_acc_list[-1], train_loss_list[-1]))
    #     print('Epoch [{}/{}], Test Accuracy: {:.4f}, Test loss: {:.4f}'.format(epoch + 1, epochs, test_acc_list[-1], test_loss_list[-1]))
    #
    #
    # # print('test_loss:',test_loss_list)
    # # print('test_acc:',test_acc_list)
    # #
    # # print('train_loss:', train_loss_list)
    # # print('train_acc:', train_acc_list)


    best_acc = 0.0

    train_loss_list = []
    train_acc_list = []
    train_precision_list = []
    train_recall_list = []
    train_f1_list = []

    test_loss_list = []
    test_acc_list = []



    # train_start_time = time.time()
    # inference_times = []  # ===== 新增：存储所有推理时间 =====
    for epoch in range(epochs):
        # epoch_start_time = time.time()

        for phase in ['train', 'test']:
            epoch_accuracy = 0
            epoch_loss = 0

            if phase == 'train':
                model.train()
                loader = trainloader
            else:
                model.eval()
                loader = testloader

            for i, batch in enumerate(loader):
                # ===== 测试阶段记录推理时间 =====
                # if phase == 'test':
                    # infer_start = time.time()
                # ================================
                optimizer.zero_grad()
                data, labels = batch
                if data.shape[0] < batch_size:
                    continue
                if device:
                    data, labels = data.cuda(), labels.cuda()

                outputs = model(data)

                # # ===== 测试阶段计算推理时间 =====
                # if phase == 'test':
                #     torch.cuda.synchronize()
                #     infer_time = time.time() - infer_start
                #     print(f'   Batch {i}: 推理时间 {infer_time * 1000:.2f}ms')
                # #                     # inference_times.append(infer_time)  # ===== 新增：存入列表 =====
                # #                 # ================================

                accuracy = get_accuracy(outputs, labels, batch_size)
                loss = criterion(outputs, labels.long())

                epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
                epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_accuracy)
            else:
                test_loss_list.append(epoch_loss)
                test_acc_list.append(epoch_accuracy)

            if phase == 'test' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                print(f'save best model,第{epoch + 1}轮')
                torch.save(model.state_dict(), './weight/best_model4_v0.pth')

    #     epoch_time = time.time() - epoch_start_time
    #
        print('Epoch [{}/{}], Train Accuracy: {:.4f}, Train loss: {:.4f}'.format(epoch + 1, epochs, train_acc_list[-1],
                                                                                 train_loss_list[-1]))
        print('Epoch [{}/{}], Test Accuracy: {:.4f}, Test loss: {:.4f}'.format(epoch + 1, epochs, test_acc_list[-1],
                                                                               test_loss_list[-1]))
    #     print(f'Epoch [{epoch + 1}/{epochs}], Time: {epoch_time:.2f}s')
    #
    # total_time = time.time() - train_start_time
    # print(f'\n总训练时间: {total_time:.2f}s ({total_time / 60:.2f} min)')
    #
    # # ===== 新增：打印平均推理时间 =====
    # if inference_times:
    #     avg_infer = sum(inference_times) / len(inference_times)
    #     print(f'\n平均推理时间: {avg_infer*1000:.2f}ms')
    #     print(f'每秒处理: {1/avg_infer:.2f} batches ({batch_size/avg_infer:.2f} samples/s)')
    # =================================


if __name__ == '__main__':
    import random
    import numpy as np
    SEED =42  # 每次手动改这个值   # 第一次42，第二次123，第三次2024
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    main()
