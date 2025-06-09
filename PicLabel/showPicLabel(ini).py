# 从.ini文件读取标签信息并在图像上显示
import configparser
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib

# 设置matplotlib默认字体为Noto Sans CJK系列，确保中文正常显示
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

# 文件路径
ini_path = os.path.join(os.path.dirname(__file__), '../data/data3-1/T1-SHJCL-10-01-02-03-1-L0-3.ini')
print(f"INI文件路径: {ini_path}")
img_path = os.path.join(os.path.dirname(__file__), '../data/data3-1/T1-SHJCL-10-01-02-03-1-L0-3.jpg')
print(f"图片文件路径: {img_path}")

# 读取ini文件
config = configparser.ConfigParser()
config.read(ini_path, encoding='utf-8')

# 读取图片
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"图片文件未找到: {img_path}")

# 获取矩形框数量
rect_num = int(config['FastPara']['RectNum'])

for i in range(rect_num):
    section = f'RectPara-{i}'
    if section not in config:
        continue
    x = int(config[section]['XValue'])
    y = int(config[section]['YValue'])
    w = int(config[section]['WidthValue'])
    h = int(config[section]['HeightValue'])
    part_name = config[section].get('PartName', f'Part-{i}')
    # 绘制矩形框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 绘制标签
    cv2.putText(img, part_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# BGR转RGB用于matplotlib显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title('标签显示结果')
plt.show()