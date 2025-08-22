# 0108整合像素中的rgb分离，并且加上单色度调整；
import os
from PIL import Image, ImageDraw

def rgb_dis(input_image_path, output_image_path, scale_factor=6):
    """
    将图像中的每个像素点转换为RGB子像素以正方形排列的方式，并按指定比例放大图像。
    蓝色子像素覆盖正方形的一半，红色和绿色各占四分之一。

    参数:
    - input_image_path: 输入图像的路径。
    - output_image_path: 输出图像的路径。
    - scale_factor: 缩放比例因子（默认为10）。
    """
    try:
        # 打开并转换图像为RGB模式
        img = Image.open(input_image_path).convert('RGB')
        width, height = img.size

        # 计算输出图像的大小
        # 每个像素将被转换为一个正方形，边长为scale_factor
        output_width = width * scale_factor
        output_height = height * scale_factor

        # 创建新的图像
        separated_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
        draw = ImageDraw.Draw(separated_img)

        for y in range(height):
            for x in range(width):
                # 获取当前像素的RGB值
                r, g, b = img.getpixel((x, y))

                # 计算该像素在输出图像中的左上角坐标
                top_left_x = x * scale_factor
                top_left_y = y * scale_factor

                # 定义每个子像素的区域
                # 蓝色覆盖正方形的下半部分
                blue_bbox = [
                    (top_left_x, top_left_y + (scale_factor // 2)),
                    (top_left_x + scale_factor, top_left_y + scale_factor)
                ]
                draw.rectangle(blue_bbox, fill=(0, 0, b))

                # 红色覆盖左上四分之一
                red_bbox = [
                    (top_left_x, top_left_y),
                    (top_left_x + scale_factor // 2, top_left_y + (scale_factor // 2))
                ]
                draw.rectangle(red_bbox, fill=(r, 0, 0))

                # 绿色覆盖右上四分之一
                green_bbox = [
                    (top_left_x + scale_factor // 2, top_left_y),
                    (top_left_x + scale_factor, top_left_y + scale_factor // 2)
                ]
                draw.rectangle(green_bbox, fill=(0, g, 0))

        # 保存处理后的图像
        separated_img.save(output_image_path)
        print(f"我是绘图，我已经将rgb分成正方形了: {output_image_path}")

    except Exception as e:
        print(f"处理图像时出错: {e}")

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


def batch_process(input_folder, output_folder, r_factor, g_factor, b_factor,scale_factor=6):
    """
    批量处理指定文件夹中的所有图像。

    参数:
    - input_folder: 输入图像文件夹路径。
    - output_folder: 输出图像文件夹路径。
    - scale_factor: 缩放比例因子（默认为10）。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_rgb_square{ext}"
            output_path = os.path.join(output_folder, output_filename)
            rgb_dis(input_path, output_path, scale_factor)
            increase_brightness(output_path, output_path, r_factor, g_factor, b_factor)


if __name__ == "__main__":
    # 用户可以在此处设置参数
    input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\test_320"  # 输入文件夹路径
    output_folder = r"F:\FPGA\unilumin_work\pic\0107\dis_4"  # 输出文件夹路径
    scale_factor = 6                  # 缩放比例因子，用户可自行调整
    # 颜色增益通道
    r_factor = 1.4  # 红色通道的增益系数
    g_factor = 1.3  # 绿色通道的增益系数
    b_factor = 1.2  # 蓝色通道的增益系数
    batch_process(input_folder, output_folder,r_factor, g_factor, b_factor, scale_factor)


