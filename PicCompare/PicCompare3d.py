# PicCompare3d.py
# 提取两幅图对应标注框区域的深度/三维点云信息，通过DBSCAN方法聚类剔除背景和离散点，只保留螺栓的点云数据
# 比较判断螺栓等目标是否松动
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

# 聚类滤波函数
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

# 5. 区域点云提取函数
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

# --------------------------------------------------
# 6. 点云粗配准
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals()
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500)
    )
    return result

# 7. 点云icp配准并比较重叠区域
def compare_region(pts1_np, pts2_np, voxel_size=0.0010, distance_thresh=0.0015, visualize=True):
    pts1_np = pts1_np * depth_unit / 1000.0 # 单位转换：从0.1毫米变为米
    pts2_np = pts2_np * depth_unit / 1000.0
    print(np.min(pts1_np, axis=0))
    print(np.max(pts1_np, axis=0))

    # 创建点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1_np)
    pcd2.points = o3d.utility.Vector3dVector(pts2_np)

    # 打印基本信息
    print(f"pcd1 点数: {len(pcd1.points)}, pcd2 点数: {len(pcd2.points)}")

    # 预处理：下采样+法线+FPFH
    pcd1_down, fpfh1 = preprocess_point_cloud(pcd1, voxel_size)
    pcd2_down, fpfh2 = preprocess_point_cloud(pcd2, voxel_size)

    # 全局粗配准（FPFH + RANSAC）
    print("开始全局配准（RANSAC）...")
    init_result = global_registration(pcd2_down, pcd1_down, fpfh2, fpfh1, voxel_size)
    print("初始配准矩阵：\n", init_result.transformation)
    if init_result.fitness < 0.1:
        print("⚠️ RANSAC 粗配准失败，fitness 过低。")
    pcd2.transform(init_result.transformation)

    # ICP 精配准
    print("开始 ICP 精配准...")
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, distance_thresh, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print("ICP 配准矩阵：\n", icp_result.transformation)
    print(f"ICP fitness: {icp_result.fitness:.3f}, inlier_rmse: {icp_result.inlier_rmse:.3f}")
    if icp_result.fitness < 0.1:
        print("⚠️ ICP 配准效果差，可能未成功重叠。")
    pcd2.transform(icp_result.transformation)

    # 构建 KDTree，统计重叠区域
    print("分析重叠区域...")
    target_tree = o3d.geometry.KDTreeFlann(pcd2)
    distances = []
    matched_points = 0
    for pt in pcd1.points:
        [_, _, dists] = target_tree.search_knn_vector_3d(pt, 1)
        dist = np.sqrt(dists[0])
        if dist < distance_thresh:
            distances.append(dist)
            matched_points += 1

    if matched_points > 0:
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)
        missing_ratio = 1 - matched_points / len(pcd1.points)
    else:
        avg_dist = float('inf')
        max_dist = float('inf')
        missing_ratio = 1.0

    print(f"✅ 重叠点数: {matched_points}/{len(pcd1.points)}")
    print(f"区域均值距离: {avg_dist:.2f}, 最大距离: {max_dist:.2f}, 缺失点比例: {missing_ratio * 100:.2f}%")

    # 可视化
    if visualize:
        pcd1.paint_uniform_color([0, 1, 0])     # green
        pcd2.paint_uniform_color([1, 0.706, 0]) # yellow
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name="配准后点云", width=800, height=600)

    return avg_dist, max_dist, missing_ratio

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
    pts1 = extract_box_points(box1, points1) # 单位0.1毫米
    pts2 = extract_box_points(box2, points2)

    compare_region(pts1, pts2, voxel_size=0.0015, distance_thresh=0.0030, visualize=True)