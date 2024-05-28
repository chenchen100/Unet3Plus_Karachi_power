import os

from PIL import Image
from tqdm import tqdm

from model import Model
from utils.utils_metrics import compute_mIoU, show_results
from utils.utils import logger
from settings import *

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    logger = logger("get_mIoU")

    image_ids = open(val_dataset, 'r').read().splitlines()

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        logger.log("Load model:")
        unet = Model()

        logger.log("Get predict result:")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(dataset, origin_jpg_img, image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))

    if miou_mode == 0 or miou_mode == 2:
        logger.log("Get miou:")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(segment_img, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        logger.log("Show_results:")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
