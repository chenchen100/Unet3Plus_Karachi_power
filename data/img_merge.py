# import os
#
# from PIL import Image
#
#
# def merge_images(image_folder, output_file, n, m):
#     # 获取所有图像文件的列表
#     image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
#     image_files.sort(key=lambda x: (int((x.split("_")[3])), int(x.split("_")[4].split(".")[0])))
#     # PAK_Karachi_0_0.jpg
#
#
#     # 计算每个小图像的大小和大图像的大小
#     image_count = len(image_files)
#     print(image_count)
#     if image_count == 0:
#         print('No image files found in the directory:', image_folder)
#         return
#
#     # 计算小图像的大小以及大图像的大小
#     img = Image.open(image_files[0])
#     img_size0 = img.size[0]
#     img_size1 = img.size[1]
#     new_img_size0 = img_size0 * n
#     new_img_size1 = img_size1 * m
#     print("size0:", img_size0, "size1:", img_size1)
#     print(new_img_size0, "newSize1",new_img_size1)
#
#     # 创建一个新的大图像
#     new_img = Image.new('RGB', (new_img_size0, new_img_size1), 'white')
#
#     # 将所有小图像粘贴到新图像的正确位置
#     for i, f in enumerate(image_files):
#         row = int(i / n)
#         col = i % n
#         img = Image.open(f)
#         img = img.resize((img_size0, img_size1))
#         new_img.paste(img, (col * img_size0, row * img_size1))
#
#     # 保存大图像
#     new_img.save(output_file)
#
#
# # 用法示例
# image_folder = 'img_out'
# output_file = 'output1.jpg'
# n = 117  # 每行显示的图像数
# m = 200  # 每列显示的图像数
# merge_images(image_folder, output_file, n, m)

import os
from PIL import Image


def merge_images(image_folder, output_file, n, m):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: (int((x.split("_")[5])), int(x.split("_")[6].split(".")[0])))

    img = Image.open(image_files[0])
    img_size0 = img.size[0]
    img_size1 = img.size[1]

    # 计算左下角方块的大小
    new_img_size0 = img_size0 * n
    new_img_size1 = img_size1 * m
    print("size0:", img_size0, "size1:", img_size1)
    print(new_img_size0, "newSize1", new_img_size1)

    new_img = Image.new('RGB', (new_img_size0, new_img_size1), 'white')

    # 只合并117-217列和100-200行的图像
    for i in range(151, 201):
        for j in range(0, 50):
            #print(i, j)
            f = f"predicted_sub_jpg_img\PAK_Karachi_{i}_{j}.jpg"
            #print(f)
            row = i - 151
            #print(row)
            col = j
            #print(col)
            img = Image.open(f)
            img = img.resize((img_size0, img_size1))
            new_img.paste(img, (col * img_size0, row * img_size1))

    new_img.save(output_file)


# 用法示例
image_folder = 'predicted_sub_jpg_img'
output_file = 'output.jpg'
n = 50  # 每行显示的图像数
m = 50  # 每列显示的图像数
merge_images(image_folder, output_file, n, m)
