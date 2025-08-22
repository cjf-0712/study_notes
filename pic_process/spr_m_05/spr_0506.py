import numpy as np
from PIL import Image
import os
import cv2


def process_image(image_path, output_path):
    # 读取图片并转换为numpy数组
    img = Image.open(image_path)
    img_array = np.array(img)  # 形状为(H, W, 3)，数据类型uint8

    # 定义3x3卷积核，用于计算周围像素的平均值
    kernel = np.ones((3, 3), dtype=np.float32) / 8.0
    kernel[1, 1] = 0.0  # 中心像素权重为0

    # 处理每个颜色通道
    for c in range(3):
        channel = img_array[:, :, c].astype(np.float32)
        # 计算周围8个像素的平均值
        avg_surround = cv2.filter2D(channel, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
        # 比较并替换小于平均值的像素
        mask = channel < avg_surround
        channel[mask] = avg_surround[mask]
        # 转换回uint8并确保值在0-255范围内
        img_array[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # 保存处理后的图片
    result_img = Image.fromarray(img_array)
    result_img.save(output_path)


def process_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查是否为图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)
            print(f"处理完成: {filename}")


if __name__ == "__main__":
    input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\test_160"  # 替换为输入文件夹路径
    output_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\final\buse_0506_2"  # 替换为输出文件夹路径
    process_folder(input_folder, output_folder)
