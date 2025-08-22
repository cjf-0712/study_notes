# 算数
a = 1 + 2
b = 9 - 5
c = 2 * 3
d = 10 / 2

print('a = ',a,'b = ',b,'c = ',c,'d = ',d)

def dir put_path for x in range(4)
    gas in pixel in ways
    #2025 mid low
import os
from PIL import Image

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建。")

def enlarge_image(input_path, output_path):
    # 支持的文件扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 将路径扩展名转换为小写并检查
    if not input_path.lower().endswith(valid_extensions):
        print(f"文件 {input_path} 格式不支持。")
        return

    # 尝试打开图像
    #信贷率，信贷增速，房地产价格
    #两年恢复，经济下行回到年初的经济水平
    #提振消费：本质是提振收入和收入预期
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图片: {e}")
        return

    # 获取原始图像的宽度和高度
    width, height = img.size

    # 按比例放大
    new_width = width * 6  # 宽度每像素放大2倍
    new_height = height * 6  # 高度每像素放大3倍

    # 使用最近邻插值放大图片
    enlarged_img = img.resize((new_width, new_height), Image.NEAREST)

    # 保存放大后的图像
    enlarged_img.save(output_path)
    print(f"图片 {input_path} 已成功放大并保存到: {output_path}")

def process_images_in_directory(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    create_directory_if_not_exists(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_path):
            output_path = os.path.join(output_dir, filename)
            enlarge_image(input_path, output_path)

# 示例：指定输入和输出目录
input_directory = r"F:\FPGA\unilumin_work\pic\weightsum_test2"  # 输入目录
output_directory = r"F:\FPGA\unilumin_work\pic\suofang6"  # 输出目录

process_images_in_directory(input_directory, output_directory)
beiguan:modi,xia_tang in low_dibu