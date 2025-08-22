import tempfile
from PIL import Image
import numpy as np
import os


# 图像卷积
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
    print(f"平滑后的图像已保存到: {output_path}")


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



def process_image(image_path, temp_path):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
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

                pixels[x, y] = (r, g, b)

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
            resized_img.save(output_path)
            print(f"我是缩放，已保存放大后的图像: {output_path}")
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")


# main def IDE two def and gen tem_path
def main(input_directory, output_directory, r_factor, g_factor, b_factor, size, kernel_weight):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)

        # 如果不是文件（可能是子目录），则跳过
        if not os.path.isfile(input_path):
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file1, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file2, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file3, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file4:
            temp_path1 = temp_file1.name
            temp_path2 = temp_file2.name
            temp_path3 = temp_file3.name
            temp_path4 = temp_file4.name
        try:
            output_path = os.path.join(output_directory, filename)
            smooth_image(input_path, temp_path1, kernel_size=size, kernel_weights=kernel_weight, border_mode='reflect')
            process_image(temp_path1, temp_path2)
            increase_brightness(temp_path2, temp_path3, r_factor, g_factor, b_factor)
            enlarge_image(temp_path3, output_path, width_scale=6, height_scale=6)
        finally:
            if os.path.exists(temp_path1):
                os.remove(temp_path1)
                print(f"已删除tempfile1")
            if os.path.exists(temp_path2):
                os.remove(temp_path2)
                print(f"已删除tempfile2")
            if os.path.exists(temp_path3):
                os.remove(temp_path3)
                print(f"已删除tempfile3")

if __name__ == "__main__":
    # input_directory = r"F:\FPGA\unilumin_work\pic\suofang1600_2"
    input_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\test_160"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\pic_ku\spr1_0427"  # 输出文件夹路径
    r_factor = 1.5
    g_factor = 1.5
    b_factor = 1.5
    size = 3
    kernel_weight = [
        [1 / 4, 1 / 4, 0],
        [1 / 4, 1 / 4, 0],
        [0, 0, 0]
    ]

    # 主程序调用
    main(input_directory, output_directory, r_factor, g_factor, b_factor, size, kernel_weight)

