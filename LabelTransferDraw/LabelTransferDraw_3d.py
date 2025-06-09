# 本文件实现了将data1目录下图片的标注框通过3D点云和位姿变换，投影到data2目录下的图片上，并可视化显示。
# 步骤包括：
# 1. 读取data1的xml标注和图片，在图片上画出标注框。
# 2. 通过getPoseTranRt.py获取两张图片之间的R、t变换。
# 3. 读取点云，将标注框四角投影到3D空间，经R、t变换后再投影到data2图片上。
# 4. 在data2图片上绘制投影后的标注框。
# 5. 使用matplotlib显示两张图片的标注效果。
# 依赖：cv2、numpy、matplotlib、yaml、getPoseTranRt.py、点云文件、相机内参文件等。
# 使用前请确保相关依赖和数据文件路径正确。
# 作者：travisjiang99
# 日期：2025-05-29

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
import importlib.util
import yaml

# 设置matplotlib默认字体为Noto Sans CJK系列，确保中文正常显示
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# 修正data1_dir和data2_dir为上级目录的data1和data2
base_dir = os.path.dirname(os.path.dirname(__file__))
data1_dir = os.path.join(base_dir, 'data/data1')
data2_dir = os.path.join(base_dir, 'data/data2')
xml_path = os.path.join(data1_dir, 'LabelData1.xml')
img1_path = os.path.join(data1_dir, 'T1-SHJCL-10-01-02-03-1-L0-5.jpg')
img2_path = os.path.join(data2_dir, 'F-T1-20250408-SHJCL05004-10-01-02-03-1-L0-5.jpg')
get_pose_path = os.path.abspath(os.path.join(base_dir, '../test_workspace/getPoseTranRt/getPoseTranRt.py'))

# 修正相机内参文件路径
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/camera_intrinsics.yaml')), 'r', encoding='utf-8') as f:
    cam_cfg = yaml.safe_load(f)
fx = cam_cfg['fx']
fy = cam_cfg['fy']
cx = cam_cfg['cx']
cy = cam_cfg['cy']
depth_unit = cam_cfg.get('depth_unit', 1.0)
# print(f'相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_unit={depth_unit}')

# 1. 读取data1的xml标注
objects = []
tree = ET.parse(xml_path)
root = tree.getroot()
for obj in root.findall('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    objects.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

# 2. 在data1图片上画框
img1 = cv2.imread(img1_path)
for obj in objects:
    cv2.rectangle(img1, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0,255,0), 2)
    cv2.putText(img1, obj['name'], (obj['xmin'], obj['ymin']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# 3. 3D点变换和投影实现标注框转移
# 调用getPoseTranRt.py获得R, t
spec = importlib.util.spec_from_file_location('getPoseTranRt', get_pose_path)
getPoseTranRt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(getPoseTranRt)
R, t = getPoseTranRt.estimate_pose(img1_path, img2_path)
print('R:', R) # 3x3旋转矩阵
print('t:', t) # 3x1平移向量

img2 = cv2.imread(img2_path)
if R is not None and t is not None:
    xyz_path = os.path.join(data1_dir, 'T1-SHJCL-10-01-02-03-1-L0-5.xyz')
    width, height = img1.shape[1], img1.shape[0]
    try:
        points = np.fromfile(xyz_path, dtype=np.uint16).reshape(-1,3) # 读取点云数据
        assert points.shape[0] == width * height, "点云数据尺寸与图像尺寸不匹配"
        print('点云shape:', points.shape)
    except Exception as e:
        print('点云读取失败:', e)
        points = None
    for obj in objects:
        pts_2d = np.array([ # 四个角点的2D坐标
            [obj['xmin'], obj['ymin']],
            [obj['xmax'], obj['ymin']],
            [obj['xmax'], obj['ymax']],
            [obj['xmin'], obj['ymax']]
        ], dtype=np.int32)
        #print('投影前2D点:', pts_2d)
        pts_3d = [] # 存储投影后的3D点
        for x, y in pts_2d:
            idx = y * width + x # 计算点在点云中的索引
            if points is not None and idx < points.shape[0]:
                pt = points[idx] 
                pts_3d.append(pt) 
            else:
                print(f'点 ({x}, {y}) 在点云中未找到或索引越界')
        pts_3d = np.array(pts_3d).T # 四个角点的3D坐标，shape为(3, 4)
        #print('原始3D点:', pts_3d)
        # 变换到data2
        # R = 
        pts_3d_new = R @ pts_3d + t.reshape(3,1) # 四个角点变换后的3D坐标，shape为(3, 4)
        #print('变换后3D点:', np.round(pts_3d_new, 2))
        # 正确投影到像素平面
        pts_2d_new = [] # 存储投影后的2D点
        for i in range(pts_3d_new.shape[1]):
            X, Y, Z = pts_3d_new[:, i]
            X = X - 30000
            Y = Y - 30000
            Z = Z 
            #print(f'点 {i} X, Y, Z: ({X:.2f}, {Y:.2f}, {Z:.2f})')
            if Z <= 0: print(f'点 {i} 的Z坐标为0或负值，无法投影'); continue
            u = fx * X / Z + cx # 透视投影公式，得到像素坐标
            v = fy * Y / Z + cy
            pts_2d_new.append([int(round(u)), int(round(v))]) # 投影后的四个角点的2D坐标
        pts_2d_new = np.array(pts_2d_new, dtype=np.int32).reshape(-1,1,2) 
        #print('投影后2D点:', pts_2d_new.reshape(-1,2))
        cv2.polylines(img2, [pts_2d_new], isClosed=True, color=(255,0,0), thickness=2)
        cv2.putText(img2, obj['name'], tuple(pts_2d_new[0,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
else:
    print('未获得有效的R,t，无法投影标注')

# 显示结果
plt.figure(figsize=(16, 8))  # 设置画布尺寸，单位为英寸
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('data1标注')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('data2投影(3D变换)')
plt.axis('off')
plt.tight_layout()  # 自动调整子图参数，填满整个画布
plt.show()
