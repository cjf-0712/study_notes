import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import tempfile

from pic_process.sub_0421.buse_0429_2 import process_folder


def calculate_complement(color, channel, intensity=0.5):
    """计算补色，考虑强度因素"""
    # 简单补色算法：255减去当前值
    complement = 255 - color
    # 应用强度因子
    return int(color * (1 - intensity) + complement * intensity)


def process_pixel(img, y, x, channel_weights=(0.3, 0.6, 0.1)):
    """处理单个像素的3x3邻域"""
    # 获取3x3邻域
    neighborhood = img[max(0, y - 1):y + 2, max(0, x - 1):x + 2]

    # 如果靠近边缘，调整邻域大小
    if neighborhood.shape[0] < 3 or neighborhood.shape[1] < 3:
        return img[y, x]

    # 计算各通道的统计量
    r = neighborhood[:, :, 0]
    g = neighborhood[:, :, 1]
    b = neighborhood[:, :, 2]

    # 计算通道差异度（标准差）
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)

    # 计算通道对比度（最大值-最小值）
    r_contrast = np.max(r) - np.min(r)
    g_contrast = np.max(g) - np.min(g)
    b_contrast = np.max(b) - np.min(b)

    # 综合评估各通道的重要性（可调整权重）
    channel_scores = [
        r_std * channel_weights[0] + r_contrast * (1 - channel_weights[0]),
        g_std * channel_weights[1] + g_contrast * (1 - channel_weights[1]),
        b_std * channel_weights[2] + b_contrast * (1 - channel_weights[2])
    ]

    # 确定主导通道
    dominant_channel = np.argmax(channel_scores)
    channel_variation = channel_scores[dominant_channel] / 255.0

    # 计算中心像素的补色
    center_pixel = img[y, x]
    new_pixel = np.copy(center_pixel)

    # 只在主导通道应用补色,主色道是选出来的
    new_pixel[dominant_channel] = calculate_complement(
        center_pixel[dominant_channel],
        dominant_channel,
        min(0.8, channel_variation * 1.5)  # 控制强度
    )

    # 可选：调整其他通道作为补偿
    # 这里简单示例：稍微增强其他通道
    other_channels = [0, 1, 2]
    other_channels.remove(dominant_channel)
    for ch in other_channels:
        new_pixel[ch] = min(255, int(center_pixel[ch] * 1.05))

    return new_pixel


def optimize_image(img):
    """优化图像处理函数"""
    height, width, _ = img.shape
    optimized_img = np.zeros_like(img)

    # 使用滑动窗口处理每个像素
    for y in tqdm(range(height), desc="处理行"):
        for x in range(width):
            optimized_img[y, x] = process_pixel(img, y, x)

    return optimized_img


def process_pic(image_path, output_dir):
    """处理单个图像"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 转换为RGB格式处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 优化图像
    optimized_img = optimize_image(img_rgb)

    # 保存结果
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"optimized_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(optimized_img, cv2.COLOR_RGB2BGR))
    print(f"已处理并保存: {output_path}")



# brightness_factor
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
            print(f"我是亮度调整，亮度调整已保存处理后的图像到: {output_path}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")


# 缺陷模拟
def process_quexian(image_path, temp_path):
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
                # 获取当前像素的RGB值
                r, g, b = pixels[x, y]

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

        # 保存处理后的图片到输出路径
        img.save(temp_path)
        print(f"已保存处理后的图片到 {temp_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")


# 缩放功能函数模块
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
            print(f"我是缩放，已保存放大后的图像: {output_path}")
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")


# main def IDE two def and gen tem_path
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file1, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file2, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file3, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file4:
            temp_path1 = temp_file1.name
            temp_path2 = temp_file2.name
            temp_path3 = temp_file3.name
            temp_path4 = temp_file4.name
        try:
            # 生成输出文件路径
            output_path = os.path.join(output_directory, filename)

            # 图像处理
            process_pic(input_directory, temp_path1)

            # 模拟图像缺陷
            process_quexian(temp_path1, temp_path2)

            # 增加亮度
            increase_brightness(temp_path2, temp_path3, r_factor, g_factor, b_factor)

            # 放大图像
            enlarge_image(temp_path3, output_path, width_scale=6, height_scale=6)
        finally:
            # 删除临时文件
            if os.path.exists(temp_path1):
                os.remove(temp_path1)
                print(f"已删除tempfile1")
            if os.path.exists(temp_path2):
                os.remove(temp_path2)
                print(f"已删除tempfile2")
            if os.path.exists(temp_path3):
                os.remove(temp_path3)
                print(f"已删除tempfile3")


# mian process
if __name__ == "__main__":
    # input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\kill_0430"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\final\buse_0430"  # 输出文件夹路径
    # 设定每个颜色通道的增益系数

    r_factor = 1.5  # 红色通道的增益系数
    g_factor = 1.5  # 绿色通道的增益系数
    b_factor = 1.5  # 蓝色通道的增益系数

    """
    #颜色增益通道
    rfactor = 1.5  # 红色通道的增益系数
    gfactor = 1.5  # 绿色通道的增益系数
    bfactor = 1.5  # 蓝色通道的增益系数
    """  # 卷积矩阵参数
    # 卷积核尺寸
    size = 3
    # 矩阵参数比
    kernel_weight = [
        [1 / 4, 1 / 4, 0],
        [1 / 4, 1 / 4, 0],
        [0, 0, 0]
    ]
    """
    border = 'reflect'

    kernel_weight = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    """
    # 主程序调用
    main(input_directory, output_directory, r_factor, g_factor, b_factor)
    # main(input_directory, output_directory, r_factor, g_factor, b_factor, size, kernel_weight)