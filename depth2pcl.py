from cv2 import cv2
import numpy as np
import sys
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, median_filter


def save_ply(filepath, pcl, pcl_rgb=None):
    # ply
    if pcl_rgb is not None:
        header = ("ply\n"
                  "format ascii 1.0\n"
                  "element vertex {}\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "property uchar red\n"
                  "property uchar green\n"
                  "property uchar blue\n"
                  "end_header\n")
        f = open(filepath, "w")
        f.write(header.format(len(pcl)))
        for i in range(len(pcl)):
            f.write(" ".join([str(j) for j in pcl[i]]) + ' ' + " ".join([str(int(j)) for j in pcl_rgb[i]]) + "\n")
        f.close()
    else:
        header = ("ply\n"
                  "format ascii 1.0\n"
                  "element vertex {}\n"
                  "property float x\n"
                  "property float y\n"
                  "property float z\n"
                  "end_header\n")
        f = open(filepath, "w")
        f.write(header.format(len(pcl)))
        for i in range(len(pcl)):
            f.write(" ".join([str(j) for j in pcl[i]]) + "\n")
        f.close()


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

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
            pcl.append([i, j, depth[i][j] * int(sys.argv[3])])
    pcl = np.array(pcl)

    save_ply(output_path, pcl)

    ply = o3d.io.read_point_cloud(output_path)
    ply = ply.voxel_down_sample(voxel_size=10)
    # o3d.visualization.draw([ply])

    index = output_path.index('.')
    # print(output_path[:index] + '_down_sample' + output_path[index:])
    o3d.io.write_point_cloud(output_path[:index] + '_down_sample' + output_path[index:], ply)


if __name__ == '__main__':
    main()
