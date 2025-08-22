from PIL import Image, ImageFilter
import os
import argparse

def smooth_image(input_path, output_path):
    """
    改善图像的锯齿状（别名效应），并保存处理后的图像。

    :param input_path: 输入图像的路径
    :param output_path: 输出图像的路径
    """
    try:
        with Image.open(input_path) as img:
            # 应用平滑滤镜来减少锯齿
            smoothed_img = img.filter(ImageFilter.SMOOTH)
            # 你也可以尝试其他滤镜，例如：
            # smoothed_img = img.filter(ImageFilter.SMOOTH_MORE)
            # smoothed_img = img.filter(ImageFilter.GaussianBlur(radius=1))
            smoothed_img.save(output_path)
            print(f"已保存处理后的图像: {output_path}")
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")

def process_directory(input_dir, output_dir):
    """
    处理指定目录下的所有 PNG 图像，改善图像的锯齿状，并将结果保存到输出目录。

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    """
    if not os.path.isdir(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"无法创建输出目录 {output_dir}: {e}")
            return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_smoothed{ext}"
            output_path = os.path.join(output_dir, output_filename)
            smooth_image(input_path, output_path)

if __name__ == "__main__":
    input_directory = r"F:\FPGA\unilumin_work\pic\caitu_1600"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\SMO1"  # 输出文件夹路径

    process_directory(input_directory , output_directory)
