# getPoseTranRt 模块说明

本模块用于处理点云或坐标数据的读取与位姿变换，适用于地铁巡检机器人相关的数据处理任务。

## 目录结构
- `getPoseTranRt.py`：主脚本，实现点云/坐标文件的读取与位姿变换计算。
- `archived/PoseTran.py`：归档的位姿变换相关代码。
- `archived/ReadXYZ.py`：归档的点云/坐标文件读取代码。

## 典型输入输出
- 输入：两张图片的完整路径。
- 输出：位姿变换的旋转矩阵R和平移向量t。

## 使用示例
其他文件可以通过导入 `getPoseTranRt.py` 并调用 `estimate_pose` 函数来实现。例如：
```python
from getPoseTranRt import estimate_pose
# 调用 estimate_pose 函数
R, t = estimate_pose('/path/to/data1/xxx.jpg', '/path/to/data2/yyy.jpg')
```
这样即可在其他 Python 文件中复用该模块的功能。

