import os

import numpy as np

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Basic info and path ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# 数据集根路径
dataset = "data"
segment_img = os.path.join(dataset, 'segment_img')  # segment_img
tobe_predicted_img = "tobe_predicted_img"
origin_jpg_img = "origin_jpg_img"

dataset_segment = os.path.join(dataset, 'dataset_segment')
train_dataset = os.path.join(dataset_segment, "train.txt")
val_dataset = os.path.join(dataset_segment, "val.txt")
test_dataset = os.path.join(dataset_segment, "test.txt")

"""
miou_mode用于指定该文件运行时计算的内容:
    - miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    - miou_mode为1代表仅仅获得预测结果。
    - miou_mode为2代表仅仅计算miou。
"""
miou_mode = 0
miou_out_path = "miou_out"
detection_results_path = "detection-results"
pred_dir = os.path.join(miou_out_path, detection_results_path)

# 权值与日志文件保存的文件夹
logs = "logs"

# model_path指向logs文件夹下的权值文件
#     训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
#     验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
model_path = "logs/ep1000-loss1.141-val_loss1.140.pth"

# 训练自己的数据集必须要修改的，自己需要的分类个数+1，如2+1
num_classes = 7

# 区分的种类
name_classes = ["background", "power_plant", "nuclear_power_plants", "oil_and_LNG_terminal",
                "oil_and_LNG_storage_facilities", "offshore_photovoltaics", "land_photovoltaics"]

# 输入图片的大小，32的倍数
input_shape = [512, 512]

# ■■■■■■■■■■■■■■■■■■■■■■■■■ Training validation dataset segmentation ■■■■■■■■■■■■■■■■■■■■■
# 想要增加测试集修改trainval_percent
trainval_percent = 1

# 修改train_percent用于改变验证集的比例 8:1:1
train_percent = 0.8
test_percent = 0.1
val_percent = 0.1

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Train parameters ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# 是否使用混合精度训练，可减少约一半的显存、需要pytorch1.7.1以上
fp16 = True

# mix_type参数用于控制检测结果的可视化方式
#    mix_type = 0的时候代表原图与生成的图进行混合
#    mix_type = 1的时候代表仅保留生成的图
#    mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
mix_type = 0

"""
num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
                开启后会加快数据读取速度，但是会占用更多内存
                keras里开启多线程有些时候速度反而慢了许多
                在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
"""
num_workers = 5

"""    
权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配

如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。

当model_path = ''的时候不加载整个模型的权值。

此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。

一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
"""
train_model_path = ""

# 是否使用Cuda,没有GPU可以设置成False
cuda = True

""" 
Init_Epoch          模型当前开始的训练世代
total_epoch         模型总共训练的epoch
batch_size          模型的batch_size
"""
init_epoch = 0
total_epoch = 1000
batch_size = 2

"""
    pretrained  是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
                如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
                如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
                如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
"""
pretrained = False

# 其它训练参数：学习率、优化器、学习率下降有关
"""    
Init_lr         模型的最大学习率
                当使用Adam优化器时建议设置  Init_lr=1e-4
                当使用SGD优化器时建议设置   Init_lr=1e-2
Min_lr          模型的最小学习率，默认为最大学习率的0.01
"""
Init_lr = 1e-4
Min_lr = Init_lr * 0.01

"""
optimizer_type  使用到的优化器种类，可选的有adam、sgd
                当使用Adam优化器时建议设置  Init_lr=1e-4
                当使用SGD优化器时建议设置   Init_lr=1e-2
momentum        优化器内部使用到的momentum参数
weight_decay    权值衰减，可防止过拟合
                adam会导致weight_decay错误，使用adam时建议设置为0。
"""
optimizer_type = "adam"
momentum = 0.9
weight_decay = 0

# lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
lr_decay_type = 'cos'

# save_period     多少个epoch保存一次权值，默认每个世代都保存
save_period = 10

""" 
建议选项：
种类少（几类）时，设置为True
种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
"""
dice_loss = True

# 是否使用focal loss来防止正负样本不平衡
focal_loss = True

"""
是否给不同种类赋予不同的损失权值，默认是平衡的。
设置的话，注意设置成numpy形式的，长度和num_classes一样。
如：
num_classes = 3
cls_weights = np.array([1, 2, 3], np.float32)
"""
cls_weights = np.ones([num_classes], np.float32)
