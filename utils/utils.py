from typing import Optional

import numpy as np
from PIL import Image


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    :param image:
    :return:
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    """
    对输入图像进行resize
    :param image:
    :param size:
    :return:
    """
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    """
    获得学习率
    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def download_weights(model_dir="data/model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    url = 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


class logger:
    def __init__(self, name: str):
        self.name = f"[{name}]:"

    def log(self, *msg: object, sep: Optional[str] = " ", end: Optional[str] = "\n", ):
        print(self.name, *msg, sep=sep, end=end)
