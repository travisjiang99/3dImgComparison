# 文件路径: /home/jzp/桌面/地铁巡检机器人/SubwayRobot/test_workspace/getPoseTran/ReadXYZ.py
# 功能: 自动读取data1或data2文件夹下的第一张jpg图片，推断对应xyz点云文件，结合camera_intrinsics.yaml中的图像参数和depth_unit，重构点云并可视化。
# 得到的points包含每个点的x, y, z坐标，单位mm。相对于相机坐标系。
# 依赖: numpy, open3d, opencv-python, pyyaml
# 使用方法: 直接运行本脚本，无需手动修改参数，确保相关文件夹和配置文件存在。
# 作者: travisjiang99
# 日期: 2025-05-29
# -----------------------------

import numpy as np
import open3d as o3d
import yaml
import cv2
import os

def read_camera_intrinsics(yaml_path=None):
    if yaml_path is None:
        # 修正：直接定位到SubwayRobot/config/camera_intrinsics.yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        yaml_path = os.path.join(base_dir, 'config', 'camera_intrinsics.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    width = data['width']
    height = data['height']
    depth_unit = data.get('depth_unit', 1.0)
    return width, height, depth_unit

# ===== 配置图像参数 =====
width, height, depth_unit = read_camera_intrinsics()

# 读取文件夹中的第一张jpg图片
def read_first_image(folder):
    # 获取test_workspace目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder_path = os.path.join(base_dir, 'data', folder)
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return None, None
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            return img, img_path
    return None, None

# 选择data1或data2中的图片
img, img_path = read_first_image('data1')
if img is None:
    img, img_path = read_first_image('data2')
if img is None:
    print('未找到图片')
    exit(1)

print(f"读取的图片路径: {img_path}")

# ===== 从二进制文件读取并重构点云数据 =====
# 自动推断xyz文件名
xyz_path = img_path.replace('.jpg', '.xyz')
if not os.path.exists(xyz_path):
    print(f"未找到对应的xyz文件: {xyz_path}")
    exit(1)

# 每个点包含3个uint16数值：x, y, z
point_data = np.fromfile(xyz_path, dtype=np.uint16)
assert point_data.size == width * height * 3, "数据尺寸不匹配，检查图像尺寸是否正确"

# 重塑为二维数组，形状为 (N, 3)
points = point_data.reshape((height * width, 3)).astype(np.float32)

# 单位换算（根据yaml配置）
# points *= depth_unit # 将单位转换为毫米

# ===== 创建 Open3D 点云对象 =====
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可选：移除z为0的无效点
# pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] > 0)[0])

# 输出3D点坐标
'''
for point in np.asarray(pcd.points):
    print(f"x={point[0]}, y={point[1]}, z={point[2]}")
'''

# 保存为文本文件
output_file = "output_points.txt"
np.savetxt(output_file, np.asarray(pcd.points), fmt='%d', comments='')

# ===== 显示点云 =====
#o3d.visualization.draw_geometries([pcd], window_name="XYZ 点云展示")
