import os
from osgeo import gdal
import numpy as np
from pathlib import Path as Path


class TifCrop:
    def __init__(self, infile, crop_size, save_path, repete_rate=0):
        """
        遥感影像分块函数
        :param infile:  输入tif文件
        :param crop_size: 分块大小，单值或元祖，int型。eg:200表示以 200*200个像元大小的方形进行分块，(100,200)表示以 100*200个像元大小的矩形进行分块
        :param save_path:
        :param repete_rate: 重复率, float, 其中值的范围为[0,1)之间, 默认值为0
        """

        self.projection = None
        self.infile = infile
        self.crop_size = crop_size
        self.save_path = save_path
        self.repete_rate = repete_rate

        # crop_size 参数判断
        if not isinstance(crop_size, int):
            if not isinstance(crop_size, tuple):
                raise Exception('crop_size 输入参数错误')
            else:
                if not (isinstance(crop_size[0], int) and isinstance(crop_size[1], int)):
                    raise Exception('crop_size 输入参数错误')

        # repete_rate 参数判断
        if repete_rate >= 1 or repete_rate < 0:
            raise Exception('repete_rate 输出参数错误')

    def crop_tif(self):
        if isinstance(self.crop_size, tuple):
            crop_size_r = self.crop_size[0]
            crop_size_c = self.crop_size[1]
        else:
            crop_size_r = self.crop_size
            crop_size_c = self.crop_size

        repete_size_r = int(crop_size_r * (1 - self.repete_rate))
        repete_size_c = int(crop_size_c * (1 - self.repete_rate))

        ds = gdal.Open(self.infile)
        data = ds.ReadAsArray()

        geotrans = ds.GetGeoTransform()
        self.projection = ds.GetProjection()

        # 将单波段影像添加一个维度
        if len(data.shape) == 2:
            data = np.array([data])

        channel, rows, cols = data.shape

        # 向上取整
        col_num = int(np.ceil(cols / repete_size_c))
        row_num = int(np.ceil(rows / repete_size_r))

        # 循环读取
        # 边缘部分按照向前扩充原则进行提取

        # 当重复率较高或分块尺寸较小时，遇到边缘部分可能存在分割相同的情况，故用以下参数进行判断避免该情况发生
        start_point = (-1, -1)

        for i in range(col_num):
            for j in range(row_num):
                row_s = repete_size_r * j
                row_e = repete_size_r * j + crop_size_r

                # 是否超出边界判断
                if row_e > rows:
                    row_s = rows - crop_size_r
                    row_e = rows

                col_s = repete_size_c * i
                col_e = repete_size_c * i + crop_size_c

                # 是否超出边界判断
                if col_e > cols:
                    col_s = cols - crop_size_c
                    col_e = cols

                data_crop = data[:, row_s:row_e, col_s:col_e]

                # 判断输出内容是否与之前存在重复情况，非完全重叠部分再进行分块输出
                if (row_s, col_s) != start_point:
                    start_point = (row_s, col_s)
                    # 地理信息存放
                    new_geotrans = (
                        geotrans[0] + geotrans[1] * col_s, geotrans[1], geotrans[2], geotrans[3] + geotrans[5] * row_s,
                        geotrans[4], geotrans[5])

                    # 输出文件名称
                    out_file = self.save_path + os.sep + Path(self.infile).stem + '_' + str(j) + '_' + str(i) + Path(
                        self.infile).suffix

                    self.tif_write(data_crop, new_geotrans, out_file)

    def tif_write(self, data, trans, ofile):
        """
        tif写入
        :param data: 分块后数组
        :param trans: 更新后的geotransform，包括六参数
        :param ofile: 输出全路径
        :return: None
        """
        # 数据类型获取
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 输出tif文件按照单波段或多波段划分
        bands, height, width = data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(ofile, int(width), int(height), int(bands), datatype)

        if dataset is not None:
            dataset.SetGeoTransform(trans)  # 写入仿射变换参数
            dataset.SetProjection(self.projection)  # 写入投影
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i])
        del dataset


def process_images_in_folder(folder_path, output_folder_path, cropsize):
    # 获取文件夹中所有文件的路径
    files = os.listdir(folder_path)
    # 遍历文件夹中的所有文件
    for file in files:
        # 检查文件是否以.tif为后缀
        if file.endswith('.tif'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file)

            TifCrop(file_path, cropsize, output_folder_path).crop_tif()


if __name__ == '__main__':
    output_folder_path = r"PAK"
    cropsize = 512
    folder_path = r"PAK_Power_Plant/"
    total_seg = [os.path.join(folder_path, seg) for seg in os.listdir(folder_path) if seg.endswith(".tif")]
    for file_path in total_seg:
        TifCrop(file_path, cropsize, output_folder_path).crop_tif()
