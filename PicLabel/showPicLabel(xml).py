# 从.xml文件读取标签信息并在图像上显示
import configparser
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib
import xml.etree.ElementTree as ET

# 设置matplotlib默认字体为Noto Sans CJK系列，确保中文正常显示
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# 文件路径
xml_path = os.path.join(os.path.dirname(__file__), '../data/data3-2/F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.xml')
img_path = os.path.join(os.path.dirname(__file__), '../data/data3-2/F-T1-20250409-SHJCL05004-10-01-02-03-1-L0-3.jpg')

# 解析XML文件
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
    w = xmax - xmin
    h = ymax - ymin
    objects.append({'name': name, 'xmin': xmin, 'ymin': ymin, 'w': w, 'h': h})

# 读取图片
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"图片文件未找到: {img_path}")

# 绘制所有标签
for obj in objects:
    x, y, w, h, name = obj['xmin'], obj['ymin'], obj['w'], obj['h'], obj['name']
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# BGR转RGB用于matplotlib显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title('标签显示结果')
plt.show()