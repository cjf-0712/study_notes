
import os
from PIL import Image

def enlarge_image(input_path, output_path):
    # 确保路径不含额外的空格
    input_path = input_path.strip()

    # 检查文件格式是否支持
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    if not any(input_path.lower().endswith(ext) for ext in valid_extensions):
        print("不支持的图片格式。")
        return

    # 打开图片
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图片: {e}")
        return

    # 获取原始图像的宽度和高度
    width, height = img.size

    # 设置新的尺寸，目标是将宽高分别放大六倍
    new_width = width * 2  # 宽度放大2倍
    new_height = height * 3  # 高度放大3倍

    # 放大图片
    enlarged_img = img.resize((new_width, new_height), Image.NEAREST)

    # 确保输出目录存在
    create_directory_if_not_exists(output_path)

    # 保存放大的图像
    enlarged_img.save(output_path)

    print(f"图片已成功放大并保存到: {output_path}")


# 示例：指定输入和输出路径
input_image_path = r"F:\FPGA\unilumin_work\pic\weightsum_test2"  # 输入路径，注意文件名及后缀
output_image_path = r"F:\FPGA\unilumin_work\pic\output\suofang_TEST1"  # 输出路径

enlarge_image(input_image_path, output_image_path)
