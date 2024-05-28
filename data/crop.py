from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 读取图像
img = Image.open('output.jpg')

# 展示原始图像
# img.show()

# 剪裁图像
# 注意：crop()函数的参数为(left, upper, right, lower)
cropped = img.crop((0, 0, 43008, 29696))

# 展示剪裁后的图像
# cropped.show()

# 保存剪裁后的图像
cropped.save('cropped.jpg')
