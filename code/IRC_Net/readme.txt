打开IRC_Net_train_3D.py, 修改data_dir及超参数并运行
lr = 1e-4  # 学习率
epochs = 50    # 训练的epoch数
batch_size = 16
trained = False # 是否使用训练好的模型
num_classes = len(data_dir['train']) # 读取训练集的类别总数