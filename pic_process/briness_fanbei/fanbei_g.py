import os
from PIL import Image


def increase_brightness(image_path, output_path, factor=2.6):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # 确保图像是RGB模式
            pixels = img.load()

            for i in range(img.width):
                for j in range(img.height):
                    r, g, b = pixels[i, j]
                    if (r, g, b) != (0, 0, 0):
                        # 增加亮度，同时确保值不超过255
                        r_new = min(int(r * factor), 255)
                        g_new = min(int(g * factor), 255)
                        b_new = min(int(b * factor), 255)
                        pixels[i, j] = (r_new, g_new, b_new)
            img.save(output_path)
            print(f"已保存处理后的图像到: {output_path}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")


def process_images(input_dir, output_dir, factor=2.6):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            increase_brightness(input_path, output_path, factor)


if __name__ == "__main__":
    input_directory = r"F:\FPGA\unilumin_work\pic\xishu\dif_gray"  # 输入文件夹路径
    output_directory = r"F:\FPGA\unilumin_work\pic\xishu\buchan_g2.6"  # 输出文件夹路径

    process_images(input_directory, output_directory, factor=2.6)
