import os
from PIL import Image


def enlarge_image(input_path, output_path, width_scale=2, height_scale=3):
    """
    放大图像的宽度和高度。

    :param input_path: 输入图像的路径
    :param output_path: 输出图像的路径
    :param width_scale: 宽度放大倍数
    :param height_scale: 高度放大倍数
    """
    try:
        with Image.open(input_path) as img:
            original_size = img.size
            new_size = (original_size[0] * width_scale, original_size[1] * height_scale)
            resized_img = img.resize(new_size, Image.NEAREST)  # 使用 NEAREST 保持像素化效果
            resized_img.save(output_path)
            print(f"已保存放大后的图像: {output_path}")
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")

def process_directory(input_dir, output_dir, width_scale=2, height_scale=3):
    """
    处理指定目录下的所有 PNG 图像，并将结果保存到输出目录。

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param width_scale: 宽度放大倍数
    :param height_scale: 高度放大倍数
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
            output_filename = f"{name}_enlarged{ext}"
            output_path = os.path.join(output_dir, output_filename)
            enlarge_image(input_path, output_path, width_scale, height_scale)

if __name__ == "__main__":
    # 用户需要设置的参数
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\kill_page"
    # 例如: "C:/Users/您的用户名/Pictures/input_images"
    output_directory = r"F:\FPGA\unilumin_work\pic\1218\base_kg"
    # 例如: "C:/Users/您的用户名/Pictures/output_images"

    # 调用处理函数
    process_directory(input_directory, output_directory, width_scale=6, height_scale=6)
