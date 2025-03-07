import colorsys
import copy
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet3Plus import UNet3Plus
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils import logger
from settings import *

logger = logger("model")


class Model(object):
    """
        使用自己训练好的模型预测需要修改2个参数：model_path和num_classes，否则出现shape不匹配
    """
    _defaults = {
        "model_path": model_path,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "mix_type": mix_type,
        "cuda": cuda,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #   画框设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        #   获得模型
        self.generate()

    #   获得所有的分类
    def generate(self, onnx=False):
        self.net = UNet3Plus(n_classes=self.num_classes)
        # print(self.net)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        logger.log(f'loaded {self.model_path} model, and classes.')
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #   检测图片
    def detect_image(self, image):
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)

        #   对输入图像进行一个备份，后面用于绘图
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            #   图片传入网络进行预测
            pr = self.net(images)[0]

            #   取出每一个像素点的种类
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            #   将灰条部分截取掉
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            #   进行图片的resize
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            #   取出每一个像素点的种类
            pr = pr.argmax(axis=-1)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            #   将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))

            #   将新图与原图及进行混合
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            #   将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            #   将新图片转换成Image的形式
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_miou_png(self, image):
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            #   图片传入网络进行预测
            pr = self.net(images)[0]

            #   取出每一个像素点的种类
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            #   将灰条部分截取掉
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            #   进行图片的resize
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            #   取出每一个像素点的种类
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
