import os
import cv2
import numpy as np
from tqdm import tqdm


def calculate_complement(color, channel, intensity=1.0):
    """计算补色，考虑强度因素"""
    # 简单补色算法：255减去当前值
    complement = 255 - color
    # 应用强度因子
    return int(color * (1 - intensity) + complement * intensity)


def process_pixel(img, y, x, channel_weights=(0.3, 0.6, 0.1)):
    """处理单个像素的3x3邻域，img是像素点的意思"""
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

    # 只在主导通道应用补色，前面计算标准差之类的这么多，都是为了得出补色的通道和基准像素点颜色。
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


def process_folder(input_folder, output_folder):
    """处理文件夹中的所有图像"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有支持的图像文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到图像文件")
        return

    for filename in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)


if __name__ == "__main__":
    # 配置输入输出路径
    input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\test_160"  # 替换为你的输入文件夹
    output_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\final\buse_0506"  # 替换为输出文件夹

    # 处理文件夹
    process_folder(input_folder, output_folder)
    print("所有图像处理完成！")
