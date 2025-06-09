# PicCompare3d.py
# 提取两幅图对应标注框区域的三维点云信息，通过DBSCAN方法聚类剔除背景和离散点，只保留螺栓的点云数据并展示
# 作者：travisjiang99  日期：2025-06-06

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import yaml
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN
import math
import open3d as o3d

# 设置matplotlib默认字体为Noto Sans CJK系列，确保中文正常显示
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# 路径配置
base_dir = os.path.dirname(os.path.dirname(__file__))
data1_dir = os.path.join(base_dir, 'data/data3-2')
data2_dir = os.path.join(base_dir, 'data/data3-3')
xml1_path = os.path.join(data1_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.xml')
xml2_path = os.path.join(data2_dir, 'LabelData2.xml')
img1_path = os.path.join(data1_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.jpg')
img2_path = os.path.join(data2_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.jpg')
xyz1_path = os.path.join(data1_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.xyz')
xyz2_path = os.path.join(data2_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.xyz')

# 读取相机内参
with open(os.path.abspath(os.path.join(base_dir, '../config/camera_intrinsics.yaml')), 'r', encoding='utf-8') as f:
    cam_cfg = yaml.safe_load(f)
fx = cam_cfg['fx']
fy = cam_cfg['fy']
cx = cam_cfg['cx']
cy = cam_cfg['cy']
width = cam_cfg['width']
height = cam_cfg['height']
depth_unit = cam_cfg.get('depth_unit')

# 1. 读取data1的xml标注
objects1 = []
tree1 = ET.parse(xml1_path)
root1 = tree1.getroot()
for obj in root1.findall('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    objects1.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

# 2. 读取data2的xml标注
objects2 = []
tree2 = ET.parse(xml2_path)
root2 = tree2.getroot()
for obj in root2.findall('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    objects2.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

# 3. 显示标注框在图像上
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
for obj in objects1:
    cv2.rectangle(img1, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0,255,0), 2)
    cv2.putText(img1, obj['name'], (obj['xmin'], obj['ymin']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
for obj in objects2:
    cv2.rectangle(img2, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,0,0), 2)
    cv2.putText(img2, obj['name'], (obj['xmin'], obj['ymin']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
plt.figure(figsize=(32,16))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('data1标注')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('data2标注')
plt.axis('off')
plt.show()

# 4. 读取点云数据
points1 = np.fromfile(xyz1_path, dtype=np.uint16).reshape(-1,3)
points2 = np.fromfile(xyz2_path, dtype=np.uint16).reshape(-1,3)

# 5. 区域点云提取和比较函数
def filter_by_dbscan(points, eps=30, min_samples=30): # eps=20
    if len(points) == 0:
        return points
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    if len(set(labels)) <= 1:  # 全是噪声或一个簇
        return points
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    main_label = unique[np.argmax(counts)]
    return points[labels == main_label]

def extract_box_points(box, points):
    # box: 4x2 array，顺序为左上、右上、右下、左下
    # points: (W*H, 3)
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(box, dtype=np.int32).reshape(-1,1,2)
    cv2.fillPoly(mask, [pts], 1)
    idxs = np.where(mask.flatten() == 1)[0]
    box_points = points[idxs]
    box_points = filter_by_dbscan(box_points)  # 过滤背景和噪声点
    return box_points

# 添加显示模式设置 1: 每个目标独立窗口，2: 所有目标合并显示
show_mode = 1

# 缓存点云和标签
all_pts1 = []
all_pts2 = []
all_names = []
all_results = []

# 遍历所有标注框，提取区域点云并比较
for obj1, obj2 in zip(objects1, objects2):
    box1 = [
        [obj1['xmin'], obj1['ymin']],
        [obj1['xmax'], obj1['ymin']],
        [obj1['xmax'], obj1['ymax']],
        [obj1['xmin'], obj1['ymax']]
    ]
    box2 = [
        [obj2['xmin'], obj2['ymin']],
        [obj2['xmax'], obj2['ymin']],
        [obj2['xmax'], obj2['ymax']],
        [obj2['xmin'], obj2['ymax']]
    ]
    pts1 = extract_box_points(box1, points1)
    pts2 = extract_box_points(box2, points2)

    all_pts1.append(pts1)
    all_pts2.append(pts2)
    all_names.append(obj1['name'])

    # 显示每个目标的3D图（仅在 show_mode == 1 时显示）
    if show_mode == 1:
        fig = plt.figure(figsize=(32,16))
        ax = fig.add_subplot(projection='3d')
        if len(pts1) > 0:
            ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], s=1, c=pts1[:, 2], cmap='viridis', label=f"data1-{obj1['name']}")
        if len(pts2) > 0:
            ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], s=1, c=pts2[:, 2], cmap='plasma', label=f"data2-{obj2['name']}")
        ax.view_init(elev=90, azim=-90)
        ax.invert_yaxis()
        ax.invert_zaxis()
        ax.set_title(f"目标: {obj1['name']} 对应点云显示")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.show()

# 如果 show_mode == 2，统一显示全部目标
if show_mode == 2:
    n = len(all_names)
    n_cols = math.ceil(math.sqrt(n))         # 列数：向上取整的平方根
    n_rows = math.ceil(n / n_cols)           # 行数：向上取整
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))  # 根据子图数动态调整大小

    for i, (pts1, pts2, name) in enumerate(zip(all_pts1, all_pts2, all_names)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        if len(pts1) > 0:
            ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], s=1, c=pts1[:, 2], cmap='viridis', label=f"data1-{name}")
        if len(pts2) > 0:
            ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], s=1, c=pts2[:, 2], cmap='plasma', label=f"data2-{name}")
        ax.view_init(elev=90, azim=-90)
        ax.invert_yaxis()
        ax.invert_zaxis()
        ax.set_title(f"{name}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    plt.tight_layout()
    plt.show()