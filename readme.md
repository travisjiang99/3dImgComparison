# 3dImgComparison

`3dImgComparison` 文件夹包含RGB-D图像匹配和局部差异比较**（使用时需要更换RGB图像和对应点云文件，以及对应的相机参数）**。以下是文件夹的主要内容：

## 文件结构

- **data/**
  - 包含多个子文件夹（如 `data1/`, `data2/`, `data3-1/` 等），存储了测试数据，包括 XML 文件、INI 配置文件、图像文件（如 `.jpg`）和点云数据（如 `.xyz`）。
  - **注：深度信息的.xyz文件超过25MB未放入。**
- **getPoseTranRt/**
  - `getPoseTranRt.md`: 文档，描述了姿态变换的相关内容。
  - `getPoseTranRt.py`: 实现姿态变换的 Python 脚本。
  - `archived/`: 存档的旧版本文件。

- **LabelTransferDraw/**
  - `LabelTransferDraw_2d.py`: 用于2D方法标签转换的脚本**（重要）**。
  - `LabelTransferDraw_3d.py`: 用于3D方法标签转换的脚本。
  - `LabelTransferDraw.md`: 文档，描述了标签转换的相关内容。

- **PicCompare/**
  - `Corres3dShow.py`: 用于显示3D点云对应关系的脚本**（重要）**。
  - `PicCompare.md`: 文档，描述了图片比较的相关内容。
  - `PicCompare3d.py`: 用于3D图片比较的脚本**（最重要）**。
  - `testXYZtoPixel.py`: 测试点云到像素转换的脚本。

- **PicLabel/**
  - `INI2XML.py`: 将INI格式的标签文件转换为XML格式的脚本。
  - `PicLabel.md`: 文档，描述了图片标注的相关内容。
  - `showPicLabel(ini).py`: 用于显示INI格式标签的脚本。
  - `showPicLabel(xml).py`: 用于显示XML格式标签的脚本。

## 使用说明

- 数据文件位于 `data/` 文件夹中，可用于测试和验证算法。
- 各子文件夹中的脚本和文档提供了不同功能的实现和说明，便于开发和调试。
