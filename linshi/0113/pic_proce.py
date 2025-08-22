#集合了图像优化和模拟缺陷以及亮度调整
#12.30审查减少了生成多文件夹，并且导入了图像优化，去除了图像缺陷的加权系数功能。
import tempfile
import os
from PIL import Image
import numpy as np
# from PIL import ImageFilter

# 图像平滑
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
# 缩放功能函数模块
def enlarge_image(input_path, output_path, width_scale=6, height_scale=6):
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

#main def IDE two def and gen tem_path，缩放比例在里面传递了，不另外增加
def main(input_directory, output_directory, r_factor, g_factor, b_factor, size, kernel_weight ,border):
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
            smooth_image(input_path, temp_path1, kernel_size=size, kernel_weights=kernel_weight, border_mode= border)

            # 模拟图像缺陷
            process_image(temp_path1, temp_path2)

            # 生成输出文件路径
            output_path = os.path.join(output_directory, filename)

            # 增加亮度
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
    output_directory = r"F:\FPGA\unilumin_work\pic\0114\ide_pro_2"  # 输出文件夹路径
    # 设定每个颜色通道的增益系数
    """
    r_factor = 0.5  # 红色通道的增益系数
    g_factor = 0.5  # 绿色通道的增益系数
    b_factor = 0.5  # 蓝色通道的增益系数
    """
    #颜色增益通道
    rfactor = 1  # 红色通道的增益系数
    gfactor = 1  # 绿色通道的增益系数
    bfactor = 1  # 蓝色通道的增益系数
    # 卷积矩阵参数
     # 卷积核尺寸
    size = 3
     # 矩阵参数比
    kernel_weight = np.array ( [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])/ 16
    border = 'reflect'
    # 主程序调用
    main(input_directory, output_directory, rfactor, gfactor, bfactor, size, kernel_weight, border)

