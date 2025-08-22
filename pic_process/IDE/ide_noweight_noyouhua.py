#IDE:在整合亮度设置和模拟图像缺陷的基础上整合了缩放功能,但是并没有图像优化模块。

import tempfile
import os
from PIL import Image

#brightness_factor
def increase_brightness(image_path, output_path, r_factor, g_factor, b_factor):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # 确保图像是RGB模式
            pixels = img.load()

            for i in range(img.width):
                for j in range(img.height):
                    r, g, b = pixels[i, j]
                    if (r, g, b) != (0, 0, 0):
                        # 分别增加每个颜色通道的亮度，并确保值不超过255
                        r_new = min(int(r * r_factor), 255)
                        g_new = min(int(g * g_factor), 255)
                        b_new = min(int(b * b_factor), 255)
                        pixels[i, j] = (r_new, g_new, b_new)

            img.save(output_path)
            print(f"已保存处理后的图像到: {output_path}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
#quexian_moni


def process_image(image_path, temp_path):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        # img.size是（w,h）的元组，相当于把元组中的数据赋值给width和height

        # 确保图片的宽度和高度是4的倍数
        if width % 4 != 0 or height % 4 != 0:
            print(f"图片 {image_path} 的尺寸不是4的倍数，跳过处理。")
            return

        for y in range(height):
                for x in range(width):
                    # 计算当前像素在4x4单元中的位置
                    row_in_unit = y % 4
                    col_in_unit = x % 4

                    r, g, b = pixels[x, y]  # 获取当前像素的RGB值

                    if row_in_unit == 0:
                        # 第一行
                        if col_in_unit == 0:
                            # 移除第一列像素
                            r, g, b = 0, 0, 0
                        elif col_in_unit == 1:
                            # 保留红色像素
                            g, b = 0, 0
                        elif col_in_unit == 2:
                            # 保留绿色像素
                            r, b = 0, 0
                        elif col_in_unit == 3:
                            # 保留蓝色像素
                            r, g = 0, 0
                    elif row_in_unit == 1:
                        # 第二行
                        if col_in_unit == 2:
                            # 移除第三列像素
                            r, g, b = 0, 0, 0
                        elif col_in_unit == 0:
                            # 保留绿色像素
                            r, b = 0, 0
                        elif col_in_unit == 1:
                            # 保留蓝色像素
                            r, g = 0, 0
                        elif col_in_unit == 3:
                            # 保留红色像素
                            g, b = 0, 0
                    elif row_in_unit == 2:
                        # 第三行
                        if col_in_unit == 1:
                            # 移除第二列像素
                            r, g, b = 0, 0, 0
                        elif col_in_unit == 0:
                            # 保留红色像素
                            g, b = 0, 0
                        elif col_in_unit == 2:
                            # 保留蓝色像素
                            r, g = 0, 0
                        elif col_in_unit == 3:
                            # 保留绿色像素
                            r, b = 0, 0
                    elif row_in_unit == 3:
                        # 第四行
                        if col_in_unit == 3:
                            # 移除第四列像素
                            r, g, b = 0, 0, 0
                        elif col_in_unit == 0:
                            # 保留蓝色像素
                            r, g = 0, 0
                        elif col_in_unit == 1:
                            # 保留绿色像素
                            r, b = 0, 0
                        elif col_in_unit == 2:
                            # 保留红色像素
                            g, b = 0, 0

                    # 更新像素值
                    pixels[x, y] = (r, g, b)

            # 保存处理后的图像
        img.save(temp_path)
        print(f"已处理并保存: {temp_path}")
        print("所有图像处理完成！")
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
#main def IDE two def and gen tem_path
def main(input_directory, output_directory, r_factor, g_factor, b_factor):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)

        # 如果不是文件（可能是子目录），则跳过
        if not os.path.isfile(input_path):
            continue

        # 使用 tempfile 创建一个临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_path = temp_file.name

        try:
            # 处理图像
            process_image(input_path, temp_path)

            # 生成输出文件路径
            output_path = os.path.join(output_directory, filename)

            # 增加亮度
            increase_brightness(temp_path, output_path, r_factor, g_factor, b_factor)
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"已删除原图片")
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
            resized_img = img.resize(new_size, Image.NEAREST)
            # 使用 NEAREST 保持像素化效果,可以替换成插值处理
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
            output_filename = f"{name}_B{ext}"
            output_path = os.path.join(output_dir, output_filename)
            enlarge_image(input_path, output_path, width_scale, height_scale)
#mian process
if __name__ == "__main__":
    #input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\test_320"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\1219\fw1"  # 输出文件夹路径
    output_directory2 = r"F:\FPGA\unilumin_work\pic\0107\noyh"
    # 设定每个颜色通道的增益系数
    """
    r_factor = 0.5  # 红色通道的增益系数
    g_factor = 0.5  # 绿色通道的增益系数
    b_factor = 0.5  # 蓝色通道的增益系数
    """
    #"""
    r_factor = 1.5  # 红色通道的增益系数
    g_factor = 1.5  # 绿色通道的增益系数
    b_factor = 1.5  # 蓝色通道的增益系数
    #"""
    main(input_directory, output_directory, r_factor, g_factor, b_factor)
    process_directory(output_directory, output_directory2, width_scale=6, height_scale=6)
