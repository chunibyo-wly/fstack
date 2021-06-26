import sys, shutil
from fstack import fstack
import re
from os import listdir
from os.path import isfile, join
from cv2 import cv2
import numpy as np
import sys
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, median_filter
from depth2pcl import *


def main():
    # step 1: focal stack to depth
    mypath = sys.argv[1]
    file_list = sorted([join(mypath, f) for f in listdir(mypath) if
                        (f.endswith('jpg') or f.endswith('JPG') or f.endswith('png') or f.endswith('tif')) and isfile(
                            join(mypath, f))], key=lambda f: int(re.sub('\D', '', f)))
    image_array = []
    for i in file_list:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        img = img[int(height * 0.05):int(height * 0.95), int(width * 0.05):int(width * 0.95)].copy()
        image_array.append(img)

    # 参数调整
    fstack(image_array, blend_size=31)

    # step 2: depth to point cloud
    input_path = "out/depth.txt"
    output_path = "out/out.ply"

    # 深度图
    f = open(input_path, "r")
    lines = f.readlines()
    depth = np.array([[float(j) for j in i.strip().split(',')] for i in lines])
    depth = median_filter(depth, size=5)
    f.close()

    height, width = depth.shape

    # 点云
    pcl = []
    for i in range(height):
        for j in range(width):
            # 参数调整
            pcl.append([i, j, depth[i][j] * int(3)])
    pcl = np.array(pcl)

    save_ply(output_path, pcl)

    ply = o3d.io.read_point_cloud(output_path)
    ply = ply.voxel_down_sample(voxel_size=10)
    o3d.visualization.draw([ply])

    index = output_path.index('.')
    o3d.io.write_point_cloud(output_path[:index] + '_down_sample' + output_path[index:], ply)


if __name__ == '__main__':
    main()
