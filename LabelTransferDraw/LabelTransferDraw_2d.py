# 本脚本用于将data1中的标注框通过特征点匹配和单应变换，转移到data2图片上，并可视化结果。保存转移后的标注框信息为XML格式。
# 支持中文显示，依赖OpenCV、matplotlib、numpy等库。
# 输出：data2目录下会生成LabelData2.xml，内容为转移后的标注框信息。
# 作者：travisjiang99  日期：2025-06-03
# 用法：直接运行本脚本，确保相关图片和xml标注文件路径正确。

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

# 路径配置
base_dir = os.path.dirname(os.path.dirname(__file__))
data1_dir = os.path.join(base_dir, 'data/data3-2')
data2_dir = os.path.join(base_dir, 'data/data3-3')
xml_path = os.path.join(data1_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.xml')
img1_path = os.path.join(data1_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.jpg')
img2_path = os.path.join(data2_dir, 'F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.jpg')

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

# 3. 特征点匹配+单应变换实现标注框转移
img2 = cv2.imread(img2_path)
# 使用ORB特征
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# 取前300个好匹配
num_good_matches = min(300, len(matches))
good_matches = matches[:num_good_matches]
# 提取匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
# 计算单应矩阵和RANSAC内点
H, mask = cv2.findHomography(
    src_pts, dst_pts,
    # cv2.RANSAC,
    ransacReprojThreshold=2.0,  # 重投影误差阈值
    maxIters=4000,              # 最大迭代次数
    confidence=0.999            # 置信度
)
if H is not None:
    mask = mask.ravel().astype(bool) # RANSAC内点掩码
    print(f'RANSAC内点数: {np.sum(mask)} / {len(mask)}')
    data2_boxes = []
    for obj in objects: # 遍历data1的标注框
        box = np.array([
            [obj['xmin'], obj['ymin']],
            [obj['xmax'], obj['ymin']],
            [obj['xmax'], obj['ymax']],
            [obj['xmin'], obj['ymax']]
        ], dtype=np.float32).reshape(-1,1,2)
        box_trans = cv2.perspectiveTransform(box, H).astype(np.int32) # 转换后的标注框
        cv2.polylines(img2, [box_trans], isClosed=True, color=(255,0,0), thickness=2)
        cv2.putText(img2, obj['name'], tuple(box_trans[0,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # 保存转移后的标注框信息
        box_pts = box_trans.reshape(-1,2)
        xmin2, ymin2 = box_pts.min(axis=0)
        xmax2, ymax2 = box_pts.max(axis=0)
        data2_boxes.append({'name': obj['name'], 'xmin': int(xmin2), 'ymin': int(ymin2), 'xmax': int(xmax2), 'ymax': int(ymax2)})
    # 保存为xml格式，结构与data1一致
    import xml.dom.minidom as minidom
    doc = minidom.Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    for box in data2_boxes:
        obj_elem = doc.createElement('object')
        name_elem = doc.createElement('name')
        name_elem.appendChild(doc.createTextNode(box['name']))
        obj_elem.appendChild(name_elem)
        bndbox_elem = doc.createElement('bndbox')
        for tag in ['xmin','ymin','xmax','ymax']:
            tag_elem = doc.createElement(tag)
            tag_elem.appendChild(doc.createTextNode(str(box[tag])))
            bndbox_elem.appendChild(tag_elem)
        obj_elem.appendChild(bndbox_elem)
        annotation.appendChild(obj_elem)
    xml_save_path = os.path.join(data2_dir, 'LabelData2.xml')
    with open(xml_save_path, 'w', encoding='utf-8') as f:
        doc.writexml(f, indent='', addindent='  ', newl='\n', encoding='utf-8')
    print(f'已保存转移后的标注框到: {xml_save_path}')
else:
    print('单应矩阵估算失败，无法转移标注')

# 显示结果
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('data1标注')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('data2投影(特征点单应)')
plt.axis('off')
plt.show()
