import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def ini_to_xml(ini_path, xml_output_path, image_path):
    # 创建 XML 根节点
    root = ET.Element("annotation")
    
    # 添加图像基本信息（从 INI 的 [FastPara] 或手动指定）
    ET.SubElement(root, "filename").text = image_path.split('/')[-1]
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "2448"  # 替换为实际值或从 INI 读取
    ET.SubElement(size, "height").text = "2048"
    ET.SubElement(size, "depth").text = "3"     # 假设 RGB 图像

    # 解析 INI 文件
    with open(ini_path, 'r') as f:
        lines = f.readlines()

    # 遍历所有矩形框
    for i, line in enumerate(lines):
        if line.startswith('[RectPara-'):
            rect_params = {}
            j = i + 1
            while j < len(lines) and not lines[j].startswith('['):
                if '=' in lines[j]:
                    key, val = lines[j].strip().split('=', 1)
                    rect_params[key] = val
                j += 1

            # 提取参数
            x = int(rect_params.get('XValue', 0))
            y = int(rect_params.get('YValue', 0))
            w = int(rect_params.get('WidthValue', 0))
            h = int(rect_params.get('HeightValue', 0))
            part_name = rect_params.get('PartName', 'unknown')

            # 添加 XML 节点
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = part_name
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(x)
            ET.SubElement(bbox, "ymin").text = str(y)
            ET.SubElement(bbox, "xmax").text = str(x + w)
            ET.SubElement(bbox, "ymax").text = str(y + h)

    # 美化 XML 格式并保存
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    with open(xml_output_path, 'w', encoding='utf-8') as f:
        f.write(dom.toprettyxml(indent="  "))

    print(f"XML 文件已保存至：{xml_output_path}")

# 调用函数
ini_path = os.path.join(os.path.dirname(__file__), '../data/data3-1/T1-SHJCL-10-01-02-03-1-L0-3.ini')
xml_output_path = os.path.join(os.path.dirname(__file__), '../data/data3-1/T1-SHJCL-10-01-02-03-1-L0-3.xml')
img_path = os.path.join(os.path.dirname(__file__), '../data/data3-1/T1-SHJCL-10-01-02-03-1-L0-3.jpg')
ini_to_xml(ini_path, xml_output_path, img_path)