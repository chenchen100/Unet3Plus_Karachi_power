import cv2 as cv
import os

# # path = os.getcwd()  # 获取代码所在目录
# path = os.getcwd()  # 获取代码所在目录
# tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]  # 获取目录中所有tif格式图像列表
# for num, i in enumerate(tif_list):  # 遍历列表
#     img = cv.imread(i, -1)  # 读取列表中的tif图像
#     cv.imwrite(i.split('.')[0]+".jpg",img)    # tif 格式转 jpg 并按原名称命名
import os
import cv2 as cv
from tqdm import tqdm

# 获取当前目录下的a文件夹和b文件夹的路径
path_a = os.path.join(os.getcwd(), 'sub_img')
path_b = os.path.join(os.getcwd(), 'sub_jpg_img')

# 确保b文件夹存在，如果不存在，则创建它
if not os.path.exists(path_b):
    os.makedirs(path_b)

# 获取a文件夹中所有的tif格式图像列表
tif_list = [x for x in os.listdir(path_a) if x.endswith(".tif")]

for num, tif_file in tqdm(enumerate(tif_list)):
    # 构建tif文件的完整路径
    tif_path = os.path.join(path_a, tif_file)

    # 读取tif图像
    img = cv.imread(tif_path, -1)

    # 构建保存jpg文件的完整路径
    jpg_file = tif_file.split('.')[0] + ".jpg"
    jpg_path = os.path.join(path_b, jpg_file)

    # 将tif格式转换为jpg格式并保存
    cv.imwrite(jpg_path, img)

print("转换完成！")
