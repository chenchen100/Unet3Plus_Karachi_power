import os
from PIL import Image
from tqdm import tqdm

# 批量转换png图片为jpg格式
if __name__ == '__main__':
    segmentationOriginImage = 'origin_img'  # png图片目录
    segmentationOriginIJPEGImage = "origin_jpg_img"  # JPG图片目录
    files_list = [os.path.join(root, file) for root, dirs, files in os.walk(segmentationOriginImage) for file in files
                  if
                  file.endswith('.png')]
    for file in tqdm(files_list, desc='转换中'):
        img = Image.open(file)
        new_file = os.path.splitext(os.path.basename(file))[0] + '.jpg'
        output_path = os.path.join(segmentationOriginIJPEGImage, new_file)
        img.convert('RGB').save(output_path)
