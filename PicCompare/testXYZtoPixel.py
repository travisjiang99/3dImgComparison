# 本脚本用于验证点云(.xyz)数据与相机内参的像素投影关系，统计投影误差，并输出部分点的投影与真实像素坐标对比，便于人工核查和调试。
# 步骤包括：读取相机内参、读取点云、单位换算、坐标变换、像素投影、误差统计和边界点一致性验证。
# 适用于地铁巡检机器人视觉系统的标定与调试。
# 作者: travisjiang99  日期: 2025-06-05
# 注：很奇怪，投影得到的像素坐标与通过行号法计算的真实像素坐标误差较大，没发现是什么问题导致的。
# -------------------------------------------------------------

import numpy as np
import yaml
import os
from scipy.optimize import minimize

# 1. 读取相机内参
def read_camera_intrinsics(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    fx = data['fx']
    fy = data['fy']
    cx = data['cx']
    cy = data['cy']
    width = data['width']
    height = data['height']
    depth_unit = data.get('depth_unit', 0.1)  # 自动读取depth_unit，默认0.1
    return fx, fy, cx, cy, width, height, depth_unit

# 2. 读取点云数据（.xyz二进制文件）
def read_xyz_points(xyz_path, width, height):
    point_data = np.fromfile(xyz_path, dtype=np.uint16)
    assert point_data.size == width * height * 3, '点云数据尺寸不匹配!'
    points = point_data.reshape((height * width, 3))
    return points

if __name__ == '__main__':
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xyz_path = os.path.join(base_dir, 'data/data1/T1-SHJCL-10-01-02-03-1-L0-5.xyz')
    yaml_path = os.path.join(base_dir, '../config/camera_intrinsics.yaml')
    yaml_path = os.path.abspath(yaml_path)

    fx, fy, cx, cy, width, height, depth_unit = read_camera_intrinsics(yaml_path)
    
    points = read_xyz_points(xyz_path, width, height)

    # 单位换算 depth_unit，并转为米
    points = points * depth_unit / 1000.0

    # 输出部分原始点坐标，便于人工核查
    """
    print("原始点样例: idx\tX\tY\tZ")
    for i in range(0, len(points), max(1, len(points)//10)):
        print(f"{i}\t{points[i,0]:.4f}\t{points[i,1]:.4f}\t{points[i,2]:.4f}")
    """

    # 3. 坐标变换（将点云原点平移到相机原点，假设原点为(3.0, 3.0, 0)）
    X = points[:, 0] - 3.0
    Y = points[:, 1] - 3.0
    Z = points[:, 2]

    # 过滤掉Z过小的点，避免除法异常
    valid = Z > 0.1  # 例如0.1米
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    print(f"有效点数: {X.shape[0]} / {points.shape[0]}")

    # 4. 投影预测像素坐标
    pred_u = fx * X / Z + cx
    pred_v = fy * Y / Z + cy

    # 5. 真实像素坐标（行号法）
    idx = np.arange(points.shape[0])[valid]
    true_u = idx % width
    true_v = idx // width

    # 6. 误差统计
    du = pred_u - true_u
    dv = pred_v - true_v
    print(f"u误差均值: {np.mean(du):.2f}, 标准差: {np.std(du):.2f}")
    print(f"v误差均值: {np.mean(dv):.2f}, 标准差: {np.std(dv):.2f}")
    print(f"u误差绝对值均值: {np.mean(np.abs(du)):.2f}")
    print(f"v误差绝对值均值: {np.mean(np.abs(dv)):.2f}")
    print("idx\tpred_u\ttrue_u\tpred_v\ttrue_v\tX\tY\tZ")
    for i in range(0, len(du), max(1, len(du)//10)):
        print(f"{idx[i]}\t{pred_u[i]:.1f}\t{true_u[i]}\t{pred_v[i]:.1f}\t{true_v[i]}\t{X[i]:.4f}\t{Y[i]:.4f}\t{Z[i]:.4f}")

    # 边界点顺序一致性验证
    '''
    check_indices = [0, width-1, (height-1)*width, width*height-1, (height//2)*width + width//2]
    print("\n像素-点云顺序一致性检查:")
    print("idx\t(u,v)\t(X,Y,Z)")
    for idx in check_indices:
        u = idx % width
        v = idx // width
        Xc, Yc, Zc = points[idx]
        print(f"{idx}\t({u},{v})\t({Xc:.4f},{Yc:.4f},{Zc:.4f})")
    '''
