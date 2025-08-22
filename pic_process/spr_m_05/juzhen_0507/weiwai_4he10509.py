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

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if kernel_size % 2 == 0:
        raise ValueError("卷积核大小必须是奇数！")

    image = Image.open(input_path).convert('RGB')
    width, height = image.size
    pixels = np.array(image, dtype=np.float32)

    if kernel_weights is None:
        kernel_weights = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    else:
        kernel_weights = np.array(kernel_weights, dtype=np.float32)

    if kernel_weights.shape != (kernel_size, kernel_size):
        raise ValueError("卷积核权重的大小必须与 kernel_size 对应！")

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

    smoothed_pixels = np.zeros_like(pixels)

    # 卷积操作
    for y in range(height):
        for x in range(width):
            region = padded_pixels[y:y + kernel_size, x:x + kernel_size]

            r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
            smoothed_r = np.sum(r * kernel_weights)
            smoothed_g = np.sum(g * kernel_weights)
            smoothed_b = np.sum(b * kernel_weights)

            smoothed_pixels[y, x] = [smoothed_r, smoothed_g, smoothed_b]

    smoothed_pixels = np.clip(smoothed_pixels, 0, 255).astype('uint8')

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

        if width % 4 != 0 or height % 4 != 0:
            print(f"图片 {image_path} 的尺寸不是4的倍数，跳过处理。")
            return

        for y in range(0, height, 4):
            for x in range(0, width, 4):
                block = []
                for j in range(0, 4):
                    row = []
                    for i in range(0, 4):
                        xi = x + i
                        yj = y + j
                        if 0 <= xi < width and 0 <= yj < height:
                            row.append(pixels[xi, yj])
                        else:
                            row.append((0, 0, 0))
                    block.append(row)

                def weighted_sum_channel(group, channel, weights):

                    if len(group) != len(weights):
                        raise ValueError("像素组和权重列表的长度必须相同。")

                    weighted_sum = sum(pixel[channel] * weight for pixel, weight in zip(group, weights))
                    return int(weighted_sum)

                # 均分权重
                weight1 = [0.25, 0.25, 0.25, 0.25]
                group1 = [block[0][0], block[1][0], block[0][1], block[1][1]]
                b_avg1 = weighted_sum_channel(group1, 2, weight1)
                pixels[x + 0, y + 0] = (0, 0, b_avg1)

                r_avg1 = weighted_sum_channel(group1, 0, weight1)
                pixels[x + 0, y + 1] = (r_avg1, 0, 0)

                g_avg1 = weighted_sum_channel(group1, 1, weight1)
                pixels[x + 1, y + 0] = (0, g_avg1, 0)
                pixels[x + 1, y + 1] = (0, 0, 0)
                group2 = [block[0][2], block[1][2], block[0][3], block[1][3]]
                b_avg2 = weighted_sum_channel(group2, 2, weight1)
                pixels[x + 1, y + 2] = (0, 0, b_avg2)

                g_avg2 = weighted_sum_channel(group2, 1, weight1)
                pixels[x + 0, y + 2] = (0, g_avg2, 0)

                pixels[x + 0, y + 3] = (0, 0, 0)

                r_avg2 = weighted_sum_channel(group2, 0, weight1)
                pixels[x + 1, y + 3] = (r_avg2, 0, 0)
                group3 = [block[2][0], block[3][0], block[2][1], block[3][1]]
                r_avg3 = weighted_sum_channel(group3, 0, weight1)
                pixels[x + 2, y + 0] = (r_avg3, 0, 0)

                g_avg3 = weighted_sum_channel(group3, 1, weight1)
                pixels[x + 3, y + 1] = (0, g_avg3, 0)

                pixels[x + 3, y + 0] = (0, 0, 0)
                b_avg3 = weighted_sum_channel(group3, 2, weight1)
                pixels[x + 2, y + 1] = (0, 0, b_avg3)
                group4 = [block[2][2], block[3][2], block[2][3], block[3][3]]
                b_avg4 = weighted_sum_channel(group4, 2, weight1)
                pixels[x + 3, y + 3] = (0, 0, b_avg4)

                pixels[x + 2, y + 2] = (0, 0, 0)

                r_avg4 = weighted_sum_channel(group4, 0, weight1)
                pixels[x + 3, y + 2] = (r_avg4, 0, 0)

                g_avg4 = weighted_sum_channel(group4, 1, weight1)
                pixels[x + 2, y + 3] = (0, g_avg4, 0)

        # 保存处理后的图片到输出路径
        img.save(temp_path)
        print(f"已保存处理后的图片到 {temp_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")

#缩放功能函数模块
def enlarge_image(input_path, output_path, width_scale=2, height_scale=3):
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
            process_image(input_path, temp_path2)


            output_path = os.path.join(output_directory, filename)


            increase_brightness(temp_path2, temp_path3, r_factor, g_factor, b_factor)


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
    output_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\final\41_0509"  # 输出文件夹路径

    rfactor = 1.5
    gfactor = 1.5
    bfactor = 1.5

    main(input_directory, output_directory, rfactor, gfactor, bfactor)

