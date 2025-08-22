def calculate_complement(color, channel, intensity=1.0):
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


def process_image(image_path, output_dir):
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

# 集合了图像优化和模拟缺陷以及亮度调整
# 12.23优化多重文件夹生成问题，优化矩阵卷积参数传递集中在主程序，功能函数集成化方便调用，图像优化功能可根据需求选择添加与否
# 2025.0424采用上采样方式，滤波模版统一D=(p(x-1,y-1) p(x-1,y) p(x,y-1) p(x,y))
import tempfile
import os
from PIL import Image
import numpy as np
# from PIL import ImageFilter

#图像平滑
def smooth_image(input_path, output_path, kernel_size=3, kernel_weights=None, border_mode='reflect'):
    """
    对图像进行卷积均值滤波（平滑处理），支持自定义卷积核大小和边界处理。

    参数:
    - input_path (str): 输入图像的文件路径。
    - output_path (str): 输出平滑后图像的文件路径。
    - kernel_size (int): 卷积核的大小（必须是奇数，默认3）。
    - kernel_weights (list of list): 自定义的卷积核权重（默认为均值卷积核）。
    - border_mode (str): 边界处理模式，可选 'reflect'（反射）、'constant'（常量填充0）、'wrap'（环绕），默认 'reflect'。
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if kernel_size % 2 == 0:
        raise ValueError("卷积核大小必须是奇数！")

    # 加载图像并转换为RGB模式
    image = Image.open(input_path).convert('RGB')
    width, height = image.size
    pixels = np.array(image, dtype=np.float32)

    # 如果未指定卷积核权重，使用默认均值卷积核
    if kernel_weights is None:
        kernel_weights = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    else:
        kernel_weights = np.array(kernel_weights, dtype=np.float32)

    if kernel_weights.shape != (kernel_size, kernel_size):
        raise ValueError("卷积核权重的大小必须与 kernel_size 对应！")

    # 边界扩展模式
    pad_size = kernel_size // 2
    if border_mode == 'reflect':
        padded_pixels = np.pad(pixels, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    elif border_mode == 'constant':
        padded_pixels = np.pad(pixels, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant',
                               constant_values=0)
    elif border_mode == 'wrap':
        padded_pixels = np.pad(pixels, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='wrap')
    else:
        raise ValueError("不支持的边界模式！可选值为 'reflect', 'constant', 'wrap'。")

    # 创建空数组用于存储平滑后的像素
    smoothed_pixels = np.zeros_like(pixels)

    # 卷积操作
    for y in range(height):
        for x in range(width):
            # 提取邻域区域
            region = padded_pixels[y:y + kernel_size, x:x + kernel_size]

            # 卷积运算
            r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
            smoothed_r = np.sum(r * kernel_weights)
            smoothed_g = np.sum(g * kernel_weights)
            smoothed_b = np.sum(b * kernel_weights)

            # 赋值给新像素
            smoothed_pixels[y, x] = [smoothed_r, smoothed_g, smoothed_b]

    # 转换为8位整数类型
    smoothed_pixels = np.clip(smoothed_pixels, 0, 255).astype('uint8')

    # 保存平滑后的图像
    smoothed_image = Image.fromarray(smoothed_pixels, 'RGB')
    smoothed_image.save(output_path)
    print(f"我是卷积，卷积后的图像已保存到: {output_path}")

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
            print(f"我是亮度调整，亮度调整已保存处理后的图像到: {output_path}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
#6*6矩阵加权模拟
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
                # 定义当前4x4块的像素数据基础上，再额外增加相邻2*2的矩阵数据
                # block = [[pixels[x + i, y + j] for i in range(-1, 5)] for j in range(-1, 5)],相当于用i,j虚数来表示block
                # 中的虚拟坐标，来解决边界的问题，边界从-1到4即可，没有超过第五行；
                block = []
                for j in range(-1, 4):
                    row = []
                    for i in range(-1, 4):
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
                rig_b = (x == width - 1 or x == width - 2)
                # 是否在上边界
                top_b = (y == 0 or y == 1)
                # 是否在下边界末行
                bot_b = (y == height - 1 or height - 2)
                # 判断边界情况
                """
                if lef_b:
                    # block[0][-1] = (1,1,1),自行修该边界外的虚拟点；
                    pixels[x + 0, y + 1] = (0, g_avg1, 0)
                else:
                    pixels[x + 0, y + 1] = (0, g_avg1, 0)
                """
                # 均分权重
                weight1 = [0.25, 0.25, 0.25, 0.25]
                # 一共16个像素网格,去掉4个黑色的,有12个组合情况
                # 处理 (0,0): B1数据
                group1 = [block[-1][-1], block[-1][0], block[1][-1], block[0][0]]
                b_avg1 = weighted_sum_channel(group1, 2, weight1)
                pixels[x + 0, y + 0] = (0, 0, b_avg1)

                # 处理 (0,1): R1数据
                # 边界情况的算式
                group2 = [block[0][0], block[-1][0], block[-1][1], block[0][1]]
                r_avg1 = weighted_sum_channel(group2, 0, weight1)
                pixels[x + 0, y + 1] = (r_avg1, 0, 0)

                # 处理 (1,0): G1数据
                group3 = [block[1][0], block[0][0], block[0][-1], block[1][-1]]
                g_avg1 = weighted_sum_channel(group3, 1, weight1)
                pixels[x + 1, y + 0] = (0, g_avg1, 0)

                # 处理 (1,1): 黑色
                pixels[x + 1, y + 1] = (0, 0, 0)

                # 处理 (1,2): B2数据
                group4 = [block[1][2], block[2][2], block[2][3], block[1][3]]
                b_avg2 = weighted_sum_channel(group4, 2, weight1)
                pixels[x + 1, y + 2] = (0, 0, b_avg2)

                # 处理 (0,2): G2数据
                group5 = [block[0][2], block[0][1], block[-1][1], block[-1][2]]
                g_avg2 = weighted_sum_channel(group5, 1, weight1)
                pixels[x + 0, y + 2] = (0, g_avg2, 0)

                # 处理 (0,3): 黑色
                pixels[x + 0, y + 3] = (0, 0, 0)

                # 处理 (1,3): R2数据
                group6 = [block[1][3], block[1][2], block[0][2], block[0][3]]
                r_avg2 = weighted_sum_channel(group6, 0, weight1)
                pixels[x + 1, y + 3] = (r_avg2, 0, 0)

                # 处理 (2,0): R3数据
                group7 = [block[2][0], block[1][0], block[1][-1], block[2][-1]]
                r_avg3 = weighted_sum_channel(group7, 0, weight1)
                pixels[x + 2, y + 0] = (r_avg3, 0, 0)

                # 处理 (3,1): G3数据
                group8 = [block[3][1], block[2][1], block[2][0], block[3][0]]
                g_avg3 = weighted_sum_channel(group8, 1, weight1)
                pixels[x + 3, y + 1] = (0, g_avg3, 0)

                # 处理 (3,0): 黑色
                pixels[x + 3, y + 0] = (0, 0, 0)

                # 处理 (2,1): B3数据
                group9 = [block[2][1], block[1][1], block[1][0], block[2][0]]
                b_avg3 = weighted_sum_channel(group9, 2, weight1)
                pixels[x + 2, y + 1] = (0, 0, b_avg3)

                # 处理 (3,3): B4数据
                group10 = [block[3][3], block[2][3], block[2][2], block[3][2]]
                b_avg4 = weighted_sum_channel(group10, 2, weight1)
                pixels[x + 3, y + 3] = (0, 0, b_avg4)

                # 处理 (2,2): 黑色
                pixels[x + 2, y + 2] = (0, 0, 0)

                # 处理 (3,2): R4数据
                group11 = [block[3][2], block[2][2], block[2][1], block[3][1]]
                r_avg4 = weighted_sum_channel(group11, 0, weight1)
                pixels[x + 3, y + 2] = (r_avg4, 0, 0)

                # 处理 (2,3): G4数据
                group12 = [block[2][3], block[1][3], block[1][2], block[2][2]]
                g_avg4 = weighted_sum_channel(group12, 1, weight1)
                pixels[x + 2, y + 3] = (0, g_avg4, 0)

        # 保存处理后的图片到输出路径
        img.save(temp_path)
        print(f"已保存处理后的图片到 {temp_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")

#缩放功能函数模块
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file2, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file3:
            temp_path1 = temp_file1.name
            temp_path2 = temp_file2.name
            temp_path3 = temp_file3.name

        try:
            # 图像平滑

            # 模拟图像缺陷
            process_image(input_path, temp_path2)

            # 生成输出文件路径
            output_path = os.path.join(output_directory, filename)

            # 增加亮度
            increase_brightness(temp_path2, temp_path3, r_factor, g_factor, b_factor)

            #放大图像
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
#mian process
if __name__ == "__main__":
    #input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\test_320"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\spr1_0424"  # 输出文件夹路径
    # 设定每个颜色通道的增益系数
    """
    r_factor = 0.5  # 红色通道的增益系数
    g_factor = 0.5  # 绿色通道的增益系数
    b_factor = 0.5  # 蓝色通道的增益系数
    """
    #颜色增益通道
    rfactor = 1.5  # 红色通道的增益系数
    gfactor = 1.5  # 绿色通道的增益系数
    bfactor = 1.5  # 蓝色通道的增益系数
    """
    # 卷积矩阵参数
     # 卷积核尺寸
    size = 3
     # 矩阵参数比
    kernel_weight = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    border = 'reflect'
    """
    # 主程序调用
    main(input_directory, output_directory, rfactor, gfactor, bfactor)

