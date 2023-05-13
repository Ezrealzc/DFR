import os
import cv2
import numpy as np

from PIL import Image


# np.set_printoptions(threshold=10000000000000)
# np.set_printoptions(linewidth=10000000000000)
# np.set_printoptions(suppress=True)

#按路径读取图片
# path="C:\\Users\zc\Desktop\denoise\data_raw\\test_1962\\offshore_1627"     #修改路径
# image_list=os.listdir(path)
# print(len(image_list))
# print(image_list)
# for image in image_list:
#     image_file_path=os.path.join(path,image)
#     print(image_file_path)
#     image_raw=Image.open(image_file_path)
image_raw=Image.open(r'raw_1.jpg')
image_raw=image_raw.convert("L")   #转换成灰度图
image_raw=np.asarray(image_raw)
print(image_raw)
# image_raw.show()
image_denoising=cv2.GaussianBlur(image_raw,(5,5),3)
img = Image.fromarray(image_denoising, 'L')
img.save(r"G_denoising.jpg")
# print(image)
img.show()

