#IDE2:在整合亮度设置和模拟图像缺陷的基础上整合了缩放功能。
#12.20在原有基础上增加了拓展了边界虚拟的block像素点以支持算法计算

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
# 6*6矩阵缺陷模拟
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

        for y in range(0, height, 4):
            for x in range(0, width, 4):
                # 定义当前4x4块的像素数据基础上，再额外增加6*6的矩阵数据
                #block = [[pixels[x + i, y + j] for i in range(-1, 5)] for j in range(-1, 5)]
                block = []
                for j in range(-1, 5):
                    row = []
                    for i in range(-1, 5):
                        xi = x + i
                        yj = y + j
                        # 边界检查
                        if 0 <= xi < width and 0 <= yj < height:
                            row.append(pixels[xi, yj])
                        else:
                            # 可以选择填充黑色或其他默认值
                            row.append((0, 0, 0))
                    block.append(row)
                # 定义函数来计算平均值，channel0，1，2分别代表rgb
                def weighted_sum_channel(group, channel, weights):
                    """
                    计算指定通道的加权和。

                    :param group: 包含像素的列表，每个像素是一个可索引的序列（如元组或列表）。
                    :param channel: 指定的通道索引。
                    :param weights: 对应每个像素的权重列表。
                    :return: 加权和的整数值。
                    """
                    if len(group) != len(weights):
                        raise ValueError("像素组和权重列表的长度必须相同。")

                    weighted_sum = sum(pixel[channel] * weight for pixel, weight in zip(group, weights))
                    return int(weighted_sum)


                # 检查边界条件
                # 是否在左边界
                lef_b = (x == 0 or x == 1)
                # 是否在右边界
                rig_b = (x == width -1 or x == width - 2)
                # 是否在上边界
                top_b = (y == 0 or y == 1)
                # 是否在下边界末行
                bot_b = (y == height - 1 or height - 2)

                # 处理 (0,0): 变为黑色
                pixels[x + 0, y + 0] = (0, 0, 0)

                # 灰色权重
                weight1 = [0.22, 0.32, 0.45, 0.26, 0.37, 1, 0.45, 0.22, 0.37, 0.32]
                # 黄色权重
                weight2 = [0.32, 0.37, 0.22, 0.45, 1, 0.37, 0.26, 0.37, 0.32, 0.22]
                # 蓝色权重
                weight3 = [0.37, 0.32, 0.37, 1, 0.45, 0.22, 0.32, 0.45, 0.26, 0.22]
                # 橙色权重
                weight4 = [0.22, 0.26, 0.45, 0.32, 0.22, 0.45, 1, 0.37, 0.32, 0.37]
                # 处理 (0,1): G数据
                #边界情况的算式
                group1 = [block[0][-1], block[-1][0], block[0][0], block[1][0], block[-1][1],
                          block[0][1], block[1][1], block[2][1], block[0][2], block[1][2]]

                g_avg1 = weighted_sum_channel(group1, 1, weight1)

                # 判断是否是左边界，和是否同时是下边界
                if lef_b:
                    #block[0][-1] = (1,1,1),自行修该边界外的虚拟点；
                    pixels[x + 0, y + 1] = (0, g_avg1, 0)
                else:
                    pixels[x + 0, y + 1] = (0, g_avg1, 0)

                # 处理 (1,0): R数据
                group2 = [block[0][-1], block[1][-1], block[-1][0], block[0][0], block[1][0],
                          block[2][0], block[0][1], block[1][1], block[2][1], block[1][2]]

                r_avg = weighted_sum_channel(group2, 0, weight2)
                pixels[x + 1, y + 0] = (r_avg, 0, 0)

                # 处理 (1,1): B数据
                group3 = [block[1][-1], block[0][0], block[1][0], block[2][0], block[-1][1],
                          block[0][1], block[1][1], block[2][1], block[0][2], block[1][2]]

                b_avg = weighted_sum_channel(group3, 2, weight4)
                pixels[x + 1, y + 1] = (0, 0, b_avg)

                # 处理 (1,2): 变为黑色
                pixels[x + 1, y + 2] = (0, 0, 0)

                # 处理 (0,2): R数据
                group4 = [block[0][0], block[1][0], block[-1][1], block[0][1], block[1][1],
                          block[2][1], block[-1][2], block[0][2], block[1][2], block[0][3]]
                r_avg2 = weighted_sum_channel(group4, 0, weight3)
                pixels[x + 0, y + 2] = (r_avg2, 0, 0)

                # 处理 (0,3): B数据
                group5 = [block[0][1], block[-1][2], block[0][2], block[1][2], block[-1][3],
                          block[0][3], block[1][3], block[2][3], block[0][4], block[1][4]]
                b_avg2 = weighted_sum_channel(group5, 2, weight1)
                pixels[x + 0, y + 3] = (0, 0, b_avg2)

                # 处理 (1,3): G数据
                group6 = [block[1][1], block[0][2], block[1][2], block[2][2], block[-1][3],
                          block[0][3], block[1][3], block[2][3], block[0][4], block[1][4]]
                g_avg2 = weighted_sum_channel(group6, 1, weight4)
                pixels[x + 1, y + 3] = (0, g_avg2, 0)

                # 处理 (2,1): 变为黑色
                pixels[x + 2, y + 1] = (0, 0, 0)

                # 处理 (3,1): R数据
                group7 = [block[3][-1], block[2][0], block[3][0], block[4][0], block[1][1],
                          block[2][1], block[3][1], block[4][1], block[2][2], block[3][2]]
                r_avg3 = weighted_sum_channel(group7, 0, weight4)
                pixels[x + 3, y + 1] = (r_avg3, 0, 0)

                # 处理 (3,0): b数据
                group8 = [block[2][-1], block[3][-1], block[1][0], block[2][0], block[3][0],
                          block[4][0], block[2][1], block[3][1], block[4][1], block[3][2]]
                b_avg3 = weighted_sum_channel(group8, 2, weight2)
                pixels[x + 3, y + 0] = (0, 0, b_avg3)

                # 处理 (2,0): G数据
                group9 = [block[2][-1], block[3][-1], block[1][0], block[2][0], block[3][0],
                          block[4][0], block[1][1], block[2][1], block[3][1], block[2][2]]
                g_avg3 = weighted_sum_channel(group9, 1, weight3)
                pixels[x + 2, y + 0] = (0, g_avg3, 0)

                # 处理 (3,3): 变为黑色
                pixels[x + 3, y + 3] = (0, 0, 0)

                # 处理 (2,2): B数据
                group10 = [block[2][0], block[3][0], block[1][1], block[2][1], block[3][1],
                          block[4][1], block[1][2], block[2][2], block[3][2], block[2][3]]
                b_avg4 = weighted_sum_channel(group10, 2, weight3)
                pixels[x + 2, y + 2] = (0, 0, b_avg4)

                # 处理 (3,2): G数据
                group11 = [block[2][0], block[3][0], block[1][1], block[2][1], block[3][1],
                           block[4][1], block[2][2], block[3][2], block[4][2], block[3][3]]
                g_avg4 = weighted_sum_channel(group11, 1, weight2)
                pixels[x + 3, y + 2] = (0, g_avg4, 0)

                # 处理 (2,3): R数据
                group12 = [block[2][1], block[1][2], block[2][2], block[3][2], block[1][3],
                           block[2][3], block[3][3], block[4][3], block[2][4], block[3][4]]
                r_avg4 = weighted_sum_channel(group12, 0, weight1)
                pixels[x + 2, y + 3] = (r_avg4, 0, 0)

        # 保存处理后的图片到输出路径
        img.save(temp_path)
        print(f"已保存处理后的图片到 {temp_path}")

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
            output_filename = f"{name}_enlarged6{ext}"
            output_path = os.path.join(output_dir, output_filename)
            enlarge_image(input_path, output_path, width_scale, height_scale)
#mian process
if __name__ == "__main__":
    #input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\word_200"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\1220\t66_1"  # 输出文件夹路径
    output_directory2 = r"F:\FPGA\unilumin_work\pic\1220\t66_4"
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
