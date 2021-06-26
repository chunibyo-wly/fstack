import os
import sys, shutil

from PIL import Image
import numpy as np
from cv2 import cv2
# from line_profiler_pycharm import profile
import matplotlib.pyplot as plt


def __dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def show(a):
    cv2.imshow('a', a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize(x):
    max_, min_ = np.max(x), np.min(x)
    normalized = (x - min_) / (max_ - min_)
    return normalized


# @profile
def fstack(imgs, log_size=13, log_std=2, delta_size=31, blend_size=31, blend_std=5, log_threshold=0.0):
    """
    FSTACK 将连续焦段的一系列图片合并为一张图片

    :param imgs:
    :param log_size: LoG (laplacian of gaussian) filter 大小
    :param log_std: LoG 标准差
    :param delta_size:
    :param blend_size: 用于混合从不同焦平面获取的像素的高斯滤波器的大小
    :param blend_std:                                      的标准差
    :param log_threshold:
    :return:
    """

    # 高斯拉普拉斯算子：https://zhuanlan.zhihu.com/p/92143464 （边缘检测任务）

    # Step 1: 图片亮度一致化
    average_intensity = np.mean(imgs[0], dtype=int)
    for index, img in enumerate(imgs):
        average_intensity_current = np.mean(img, dtype=int)
        tmp = img + average_intensity - average_intensity_current
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        imgs[index] = tmp

    laplacian_path = 'out/align'
    __dir(laplacian_path)
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(laplacian_path, '{}.jpg'.format(i)), img)

    # Step 2: LoG 滤波
    ball_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (delta_size, delta_size))
    imgs_filtered = []
    for img in imgs:
        img_filtered = cv2.GaussianBlur(img, (log_size, log_size), log_std)
        img_filtered = cv2.Laplacian(img_filtered, cv2.CV_64F, ksize=log_size)
        img_filtered = cv2.dilate(img_filtered, ball_kernel, iterations=1)
        imgs_filtered.append(img_filtered)

    laplacian_path = 'out/laplacian'
    __dir(laplacian_path)
    for i, img in enumerate(imgs_filtered):
        cv2.imwrite(os.path.join(laplacian_path, '{}.jpg'.format(i)), img)

    # Step 3: 寻找对焦的像素位置
    # fmap = np.zeros(imgs[0].shape)
    # log_response = np.zeros(imgs[0].shape) + log_threshold
    # for i, img_filtered in enumerate(imgs_filtered):
    #     index = np.where(img_filtered > log_response)
    #     log_response[index] = img_filtered[index]
    #     fmap[index] = i

    fmap = np.argmax(imgs_filtered, axis=0)
    # Step 4: 平滑
    fmap = cv2.GaussianBlur(fmap.astype(np.float), (blend_size, blend_size), blend_std)
    fmap[fmap < 0] = 0

    np.savetxt('out/response.txt', fmap, delimiter=',', fmt='%1.3f')

    # Step 4: 从每张图像中提取对焦像素
    out_image = imgs[0].copy()
    for i, img in enumerate(imgs):
        index = np.where(fmap == i)
        out_image[index] = img[index]

    # Step 5: 混合焦平面
    for i in range(0, len(imgs) - 1):
        index = np.where((i < fmap) & (fmap < i + 1))
        out_image[index] = np.multiply((fmap[index] - i), imgs[i + 1][index]) + \
                           np.multiply((i + 1 - fmap[index]), imgs[i][index])

    cv2.imwrite('out/merged.jpg'.format(blend_size, log_size), out_image)
    np.savetxt('out/depth.txt'.format(blend_size, log_size), fmap, delimiter=',', fmt='%1.3f')

    fmap = 255.0 - (normalize(fmap) * 255.0)

    plt.imsave('out/depth.jpg', fmap, cmap="plasma")


def main():
    import re
    from os import listdir
    from os.path import isfile, join
    mypath = sys.argv[1]
    file_list = sorted([join(mypath, f) for f in listdir(mypath) if
                        (f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('tif')) and isfile(
                            join(mypath, f))], key=lambda f: int(re.sub('\D', '', f)))
    image_array = []
    for i in file_list:
        # pil_img = Image.open(i).convert('L')
        # img = np.array(pil_img)
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        img = img[int(height * 0.05):int(height * 0.95), int(width * 0.05):int(width * 0.95)].copy()
        image_array.append(img)

    # cv2.imshow('a', image_array[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # fstack(image_array, blend_size=31, log_threshold=0.0, log_size=31)
    fstack(image_array, blend_size=31)


# def main():
#     image_array = []
#     for i in range(1, 42):
#         pil_img = Image.open('input/{}.tif'.format(i))
#         img = np.array(pil_img)[:, :].copy()
#         image_array.append(img)
#     fstack(image_array)


if __name__ == '__main__':
    main()
