#含有优化功能和拼接功能的版本
import os
import cv2
import numpy as np


def reduce_noise(image):
    """
    使用非局部均值去噪算法减少图像噪点。
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def apply_antialiasing(image, scale_factor=1.0):
    """
    应用抗锯齿，通过调整图像尺寸实现平滑效果。
    scale_factor: 缩放因子，大于1放大，小于1缩小。
    """
    if scale_factor != 1.0:
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        # 使用双线性插值进行缩放
        return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return image


def concatenate_images(original, optimized):
    """
    拼接原图与优化后的图像进行比较。
    """
    # 确保两张图像的尺寸相同
    if original.shape != optimized.shape:
        optimized = cv2.resize(optimized, (original.shape[1], original.shape[0]))

    # 水平拼接
    concatenated = np.hstack((original, optimized))

    return concatenated


def process_image(image_path):
    """
    读取图像并应用噪点减少和抗锯齿优化，同时生成比较图像。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    #利用CV创建图像变量，通过不同的变量接力处理
    # 噪点减少
    denoised_image = reduce_noise(image)

    # 抗锯齿（例如，缩小到0.5倍再放大回原尺寸）
    scaled_down = apply_antialiasing(denoised_image, scale_factor=0.5)
    antialiased_image = apply_antialiasing(scaled_down, scale_factor=2.0)

    # 生成比较图像
    comparison_image = concatenate_images(image, antialiased_image)

    return antialiased_image, comparison_image


def process_directory(input_directory, output_directory):
    """
    处理输入目录中的所有图像，保存优化后的图像和比较图像到输出目录。
    """
    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"已创建输出目录: {output_directory}")

    # 遍历输入目录中的文件
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_directory, filename)
            optimized_path = os.path.join(output_directory, filename)

            # 生成比较图像的文件名，例如: originalname_compare.jpg
            name, ext = os.path.splitext(filename)
            comparison_filename = f"{name}_compare{ext}"
            comparison_path = os.path.join(output_directory, comparison_filename)

            print(f"正在处理: {input_path}")

            optimized_image, comparison_image = process_image(input_path)

            if optimized_image is not None:
                # 保存优化后的图像
                cv2.imwrite(optimized_path, optimized_image)
                print(f"已保存优化后的图像到: {optimized_path}")

                # 保存比较图像
                cv2.imwrite(comparison_path, comparison_image)
                print(f"已保存比较图像到: {comparison_path}")
            else:
                print(f"跳过保存: {input_path}")


if __name__ == "__main__":
    # 示例用法
    input_dir = r"F:\FPGA\unilumin_work\pic\word"  # 替换为你的输入目录路径
    output_dir = r"F:\FPGA\unilumin_work\pic\youhua_word"  # 替换为你的输出目录路径

    process_directory(input_dir, output_dir)
    print("图像处理完成。")
