# PicLabel 模块文档

## 文件说明

- `INI2XML.py`：用于将INI格式文件转换为XML格式文件的脚本。
- `showPicLabel(ini).py`：用于显示INI格式标注文件的脚本。
- `showPicLabel(xml).py`：用于显示XML格式标注文件的脚本。

## 功能描述

PicLabel模块主要用于处理和显示图片标注数据，支持INI和XML两种格式。以下是模块的主要功能：

1. **格式转换**：
   - 使用`INI2XML.py`脚本，可以将INI格式的标注文件转换为XML格式，便于后续处理和兼容性。

2. **标注显示**：
   - 使用`showPicLabel(ini).py`脚本，可以加载并显示INI格式的标注文件。
   - 使用`showPicLabel(xml).py`脚本，可以加载并显示XML格式的标注文件。

## 使用方法

1. **格式转换**：
   - 运行`INI2XML.py`脚本，输入INI文件路径，生成对应的XML文件。

   ```bash
   python3 INI2XML.py <input_ini_file> <output_xml_file>
   ```

2. **标注显示**：
   - 显示INI格式标注：

     ```bash
     python3 showPicLabel(ini).py <input_ini_file>
     ```

   - 显示XML格式标注：

     ```bash
     python3 showPicLabel(xml).py <input_xml_file>
     ```

## 注意事项

- 确保输入文件路径正确，且文件格式符合要求。
- 如果脚本运行过程中出现错误，请检查文件内容是否完整或格式是否正确。

## 示例

- 将`example.ini`转换为`example.xml`：

  ```bash
  python3 INI2XML.py example.ini example.xml
  ```

- 显示`example.ini`的标注内容：

  ```bash
  python3 showPicLabel(ini).py example.ini
  ```

- 显示`example.xml`的标注内容：

  ```bash
  python3 showPicLabel(xml).py example.xml
  ```