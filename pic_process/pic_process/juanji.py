from PIL import Image
import numpy as np
import os


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


    # 将像素数组转换为PIL图像对象
    smoothed_image = Image.fromarray(smoothed_pixels, 'RGB')
    # 保存平滑后的图像
    smoothed_image.save(output_path)
    print(f"平滑后的图像已保存到: {output_path}")
