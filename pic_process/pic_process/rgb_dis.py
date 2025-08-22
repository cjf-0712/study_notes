import os
import tempfile
from PIL import Image

def separate_rgb_channels(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 打开图像
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('RGB')

            # 创建一个新的图像用于存储RGB分离效果
            width, height = img.size
            separated_img = Image.new('RGB', (width, height), (0, 0, 0))

            for y in range(height):
                for x in range(width):
                    r, g, b = img.getpixel((x, y))

                    # 分别显示红、绿、蓝通道
                    # 创建三个独立的图像，每个图像仅包含一个通道，不同颜色亮度值
                    separated_red = (r, 0, 0)
                    separated_green = (0, g, 0)
                    separated_blue = (0, 0, b)

                    # 计算显示效果，将三种颜色叠加显示
                    new_pixel = (
                        min(r + separated_green[0] + separated_blue[0], 255),
                        min(separated_red[1] + g + separated_blue[1], 255),
                        min(separated_red[2] + separated_green[2] + b, 255)
                    )

                    # 设置像素
                    separated_img.putpixel((x, y), new_pixel)

            # 保存处理后的图像
            base_filename = os.path.splitext(filename)[0]
            separated_img.save(os.path.join(output_folder, f'{base_filename}_separated.png'))

            print(f'Processed {filename} with RGB separation.')


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
def main(input_directory, output_directory):

        # 使用 tempfile 创建一个临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file2:
            temp_path1 = temp_file1.name
            temp_path2 = temp_file2.name


        try:
            #分离颜色
            separate_rgb_channels(input_directory, temp_file1)


            #缩放
            enlarge_image(temp_file1, output_directory, width_scale=6, height_scale=6)
        finally:
            # 删除临时文件
            if os.path.exists(temp_path1):
                os.remove(temp_path1)
                print(f"已删除tempfile1")
            if os.path.exists(temp_path2):
                os.remove(temp_path2)
                print(f"已删除tempfile2")

# 使用示例
input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\word_200"  # 输入文件夹路径
output_folder = r"F:\FPGA\unilumin_work\pic\1230\1"  # 输出文件夹路径

main(input_folder, output_folder)
