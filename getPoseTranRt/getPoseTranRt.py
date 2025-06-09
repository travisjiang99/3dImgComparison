# 文件路径：/home/jzp/桌面/地铁巡检机器人/SubwayRobot/test_workspace/getPoseTran/getPoseTranRt.py
# 功能：输入两张图片路径，自动读取对应点云，输出两帧之间的真实尺度旋转矩阵R和平移向量t（单位mm）。
# 依赖: OpenCV 4, numpy 1.24.3, matplotlib 0.1.6, open3d, pyyaml
# 使用方法: 直接调用estimate_pose(img1_path, img2_path)函数，或运行本脚本进行测试

# 函数接口：estimate_pose(img1_path, img2_path)
# 函数输入：两张图片的路径
# 函数输出：旋转矩阵R（3x3）和平移向量t（3x1），可直接供其他脚本调用
# 举例：R, t = estimate_pose('/path/to/data1/xxx.jpg', '/path/to/data2/yyy.jpg')

# 作者: travisjiang99
# 日期: 2025-05-29
# -----------------------------

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import yaml
import open3d as o3d

# 读取文件夹中的第一张jpg图片
def read_first_image(folder):
    import os
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return None
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            return cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
    return None

# 读取相机内参
def read_camera_intrinsics(yaml_path=None):
    import os
    if yaml_path is None:
        # 自动定位到config目录下的camera_intrinsics.yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        yaml_path = os.path.join(base_dir, 'config', 'camera_intrinsics.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    fx = data['fx']
    fy = data['fy']
    cx = data['cx']
    cy = data['cy']
    width = data['width']
    height = data['height']
    depth_unit = data.get('depth_unit', 1.0)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, fx, fy, cx, cy, width, height, depth_unit

# 读取点云数据
def read_xyz_points_from_img(img_path, width, height, depth_unit):
    xyz_path = img_path.replace('.jpg', '.xyz')
    if not os.path.exists(xyz_path):
        print(f"未找到对应的xyz文件: {xyz_path}")
        return None
    point_data = np.fromfile(xyz_path, dtype=np.uint16)
    if point_data.size != width * height * 3:
        print("数据尺寸不匹配，检查图像尺寸是否正确")
        return None
    points = point_data.reshape((height * width, 3))
    points
    return points

def visualize_matches(img1, kp1, img2, kp2, matches, title='Feature Matches'):
    """
    可视化两张图片的特征点匹配情况。
    """
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def estimate_pose(img1_path, img2_path):
    """
    输入两张图片路径，自动读取对应点云，输出真实尺度下的旋转矩阵R和平移向量t。
    R, t均为从img1到img2的相机位姿变换。
    """
    K, fx, fy, cx, cy, width, height, depth_unit = read_camera_intrinsics()
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    points1 = read_xyz_points_from_img(img1_path, width, height, depth_unit)
    points2 = read_xyz_points_from_img(img2_path, width, height, depth_unit)
    if img1 is None or img2 is None or points1 is None or points2 is None:
        print('未找到图片或点云')
        return None, None
    # ORB特征点检测与描述
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # KNN匹配+比值测试
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = good_matches[:300]  # 限制匹配数量，避免过多内存消耗
    if len(good_matches) > 8:
        # 本质矩阵+RANSAC筛选内点
        E, mask = cv2.findEssentialMat(
            np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2),
            np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2),
            focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inlier_matches = [good_matches[i] for i in range(len(good_matches))] # if mask[i]
        print("inlier_matches | good_matches:", len(inlier_matches), "|", len(good_matches))
        # 只可视化RANSAC内点
        visualize_matches(img1, kp1, img2, kp2, inlier_matches, title='RANSAC Inlier Matches')
        if E is not None and len(inlier_matches) > 8:
            # 提取内点的三维坐标
            src_3d = []
            dst_3d = []
            for m in inlier_matches:
                idx1 = int(round(kp1[m.queryIdx].pt[1])) * width + int(round(kp1[m.queryIdx].pt[0]))
                idx2 = int(round(kp2[m.trainIdx].pt[1])) * width + int(round(kp2[m.trainIdx].pt[0]))
                if idx1 < points1.shape[0] and idx2 < points2.shape[0]:
                    p1 = points1[idx1]
                    p2 = points2[idx2]
                    if p1[2] > 0 and p2[2] > 0:
                        src_3d.append(p1)
                        dst_3d.append(p2)
            src_3d = np.array(src_3d)
            dst_3d = np.array(dst_3d)
            if len(src_3d) > 8:
                # SVD/Umeyama方法估算R, t
                src_mean = np.mean(src_3d, axis=0)
                dst_mean = np.mean(dst_3d, axis=0)
                src_centered = src_3d - src_mean
                dst_centered = dst_3d - dst_mean
                H = src_centered.T @ dst_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[2,:] *= -1
                    R = Vt.T @ U.T
                t = dst_mean - R @ src_mean
                return R, t
    return None, None

def main():
    # 示例用法：自动读取test_workspace/data/data1和test_workspace/data/data2下的第一张图片
    def get_first_img_path(folder):
        # 修正：定位到test_workspace/data/data1或data2
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 先到test_workspace根目录
        test_workspace_dir = os.path.dirname(base_dir)
        folder_path = os.path.join(test_workspace_dir, 'data', folder)
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return None
        for file in os.listdir(folder_path):
            if file.lower().endswith('.jpg'):
                return os.path.join(folder_path, file)
        return None
    img1_path = get_first_img_path('data1')
    img2_path = get_first_img_path('data2')
    if img1_path and img2_path:
        R, t = estimate_pose(img1_path, img2_path)
        if R is not None and t is not None:
            print('尺度旋转矩阵 R:')
            print(R)
            print('尺度平移向量 t:')
            print(t)
        else:
            print('未能估算出有效的位姿变换')
    else:
        print('未找到图片')

if __name__ == '__main__':
    main()
