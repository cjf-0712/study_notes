#缩放目标分辨率
import os
from PIL import Image

# 设置原始图片文件夹路径
input_folder = r'F:\FPGA\unilumin_work\pic\pic_ku\test_pic'
output_dir = r'F:\FPGA\unilumin_work\pic\pic_ku\test0425_320'
# 设置新图片文件夹路径
output_folder = os.path.join(input_folder, output_dir)

# 设置目标分辨率
new_size = (320, 320)

# 支持的图片格式
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# 创建新文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f'已创建新文件夹: {output_folder}')
else:
    print(f'新文件夹已存在: {output_folder}')

# 遍历原始文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_formats):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            with Image.open(input_path) as img:
                # 调整图片大小
                resized_img = img.resize(new_size, Image.LANCZOS)
                # 保存调整后的图片到新文件夹
                resized_img.save(output_path)
                print(f'已调整大小并保存: {filename}')
        except Exception as e:
            print(f'处理 {filename} 时出错: {e}')
