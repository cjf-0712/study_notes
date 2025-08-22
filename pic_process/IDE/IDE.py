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

        for y in range(0, height, 4):
            for x in range(0, width, 4):
                # 定义当前4x4块的像素数据
                block = [[pixels[x + i, y + j] for i in range(4)] for j in range(4)]

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
                #        weighted_sum_channel(group1, 0, weights)
                weight_r = [0.25, 0.25, 0.25, 0.25]
                weight_g = [0.25, 0.25, 0.25, 0.25]
                weight_b = [0.25, 0.25, 0.25, 0.25]
                group1 = [block[0][0], block[0][1], block[1][0], block[1][1]]
                # 处理 (0,0): 变为黑色
                pixels[x + 0, y + 0] = (0, 0, 0)

                # 处理 (0,1): G数据
                g_avg = weighted_sum_channel(group1, 1, weight_g)
                pixels[x + 0, y + 1] = (0, g_avg, 0)

                # 处理 (1,0): R数据

                r_avg = weighted_sum_channel(group1, 0, weight_r)
                pixels[x + 1, y + 0] = (r_avg, 0, 0)

                # 处理 (1,1): B数据
                b_avg = weighted_sum_channel(group1, 2, weight_b)
                pixels[x + 1, y + 1] = (0, 0, b_avg)

                # 处理 (1,2): 变为黑色
                pixels[x + 1, y + 2] = (0, 0, 0)

                # 处理 (1,3): G数据
                group2 = [block[0][2], block[0][3], block[1][2], block[1][3]]
                g_avg2 = weighted_sum_channel(group2, 1, weight_g)
                pixels[x + 1, y + 3] = (0, g_avg2, 0)

                # 处理 (0,3): B数据
                b_avg2 = weighted_sum_channel(group2, 2, weight_b)
                pixels[x + 0, y + 3] = (0, 0, b_avg2)

                # 处理 (0,2): R数据
                r_avg2 = weighted_sum_channel(group2, 0, weight_r)
                pixels[x + 0, y + 2] = (r_avg2, 0, 0)

                # 处理 (2,1): 变为黑色
                pixels[x + 2, y + 1] = (0, 0, 0)

                # 处理 (3,1): R数据
                group3 = [block[2][0], block[2][1], block[3][0], block[3][1]]
                r_avg3 = weighted_sum_channel(group3, 0, weight_r)
                pixels[x + 3, y + 1] = (r_avg3, 0, 0)

                # 处理 (3,0): b数据
                b_avg3 = weighted_sum_channel(group3, 2, weight_b)
                pixels[x + 3, y + 0] = (0, 0, b_avg3)

                # 处理 (2,0): G数据
                g_avg3 = weighted_sum_channel(group3, 1, weight_g)
                pixels[x + 2, y + 0] = (0, g_avg3, 0)

                # 处理 (3,3): 变为黑色
                pixels[x + 3, y + 3] = (0, 0, 0)

                # 处理 (2,2): B数据
                group4 = [block[2][2], block[2][3], block[3][2], block[3][3]]
                b_avg4 = weighted_sum_channel(group4, 2, weight_b)
                pixels[x + 2, y + 2] = (0, 0, b_avg4)

                # 处理 (3,2): G数据
                g_avg4 = weighted_sum_channel(group4, 1, weight_g)
                pixels[x + 3, y + 2] = (0, g_avg4, 0)

                # 处理 (2,3): R数据
                r_avg4 = weighted_sum_channel(group4, 0, weight_r)
                pixels[x + 2, y + 3] = (r_avg4, 0, 0)

        # 保存处理后的图片到输出路径
        img.save(temp_path)
        print(f"已保存处理后的图片到 {temp_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")


"""
def process_images(input_dir, output_dir, r_factor, g_factor, b_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)
            increase_brightness(input_path, output_path, r_factor, g_factor, b_factor)
"""
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

#mian process
if __name__ == "__main__":
    #input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\caitu_1600"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\IDE3"  # 输出文件夹路径

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
