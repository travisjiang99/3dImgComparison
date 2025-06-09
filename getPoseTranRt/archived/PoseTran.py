# 文件路径: /home/jzp/桌面/地铁巡检机器人/SubwayRobot/test_workspace/getPoseTran/PoseTran.py
# 功能：读取两个文件夹中的第一张jpg图片，使用ORB特征点检测与描述，KNN匹配+比值测试，RANSAC筛选内点。
#      调用cv2.recoverPose函数，利用本质矩阵E和内点特征点坐标，恢复出两张图片之间的旋转矩阵R和平移向量t。
#      融合ReadXYZ.py的点云读取与三维点提取逻辑，实现基于特征点匹配的真实尺度位姿变换输出。
# 依赖: OpenCV 4, numpy 1.24.3, matplotlib 0.1.6, open3d
# 使用方法: 确保'data1'和'data2'文件夹下各有一张jpg图片，直接运行本脚本。
# 作者: travisjiang99
# 日期: 2025-05-28
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
    # 获取test_workspace目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folder_path = os.path.join(base_dir, 'data', folder)
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return None
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            return cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
    return None

def read_camera_intrinsics(yaml_path=None):
    import os
    if yaml_path is None:
        # 修正：直接定位到SubwayRobot/config/camera_intrinsics.yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

def read_xyz_points_from_img(img_path, width, height, depth_unit):
    xyz_path = img_path.replace('.jpg', '.xyz')
    if not os.path.exists(xyz_path):
        print(f"未找到对应的xyz文件: {xyz_path}")
        return None
    point_data = np.fromfile(xyz_path, dtype=np.uint16)
    if point_data.size != width * height * 3:
        print("数据尺寸不匹配，检查图像尺寸是否正确")
        return None
    points = point_data.reshape((height * width, 3)).astype(np.float32)
    points *= depth_unit
    return points

def main():
    # 读取相机内参（自动定位config目录）
    K, fx, fy, cx, cy, width, height, depth_unit = read_camera_intrinsics()
    # 读取图片和点云
    def get_img_and_points(folder):
        # 获取test_workspace目录
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder_path = os.path.join(base_dir, 'data', folder)
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return None, None, None
        for file in os.listdir(folder_path):
            if file.lower().endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                points = read_xyz_points_from_img(img_path, width, height, depth_unit)
                return img, img_path, points
        return None, None, None

    img1, img1_path, points1 = get_img_and_points('data1')
    img2, img2_path, points2 = get_img_and_points('data2')
    if img1 is None or img2 is None or points1 is None or points2 is None:
        print('未找到图片或点云')
        return

    # ORB特征点检测与描述
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # KNN匹配+比值测试，提升匹配准确性
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"通过比值测试的匹配点数: {len(good_matches)}")

    if len(good_matches) > 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        # 用RANSAC进一步筛选内点，使用真实相机内参
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
        print(f"RANSAC内点数: {len(inlier_matches)}")
        # 可视化RANSAC内点
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=2)
        plt.figure(figsize=(16,8))
        plt.imshow(img_matches, cmap='gray')
        plt.title('Inlier Feature Matches')
        plt.axis('off')
        plt.show()
        if E is not None and len(inlier_matches) > 8:
            # ===== 真实尺度位姿变换 =====
            # 利用点云获取真实三维坐标
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
                # 使用SVD/Umeyama方法估算真实尺度的R, t
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
                print('真实尺度旋转矩阵 R:')
                print(R)
                print('真实尺度平移向量 t (单位mm):')
                print(t)
            else:
                print('有效三维点不足，无法估算真实尺度位姿变换')
        else:
            print('未能计算本质矩阵或内点不足')
    else:
        print('匹配点不足，无法计算位姿变换')

if __name__ == '__main__':
    main()
