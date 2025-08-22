import os
from PIL import Image, ImageEnhance

# 设置图片所在的目录路径
input_folder = r'F:\FPGA\unilumin_work\processed_pic'
output_folder = r'F:\FPGA\unilumin_work\processed_pic\pic_brighten15'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义支持的图片格式
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_formats):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 打开图片
            with Image.open(input_path) as img:
                # 使用ImageEnhance增强亮度
                enhancer = ImageEnhance.Brightness(img)
                # 提升亮度4倍
                brightened_img = enhancer.enhance(1.5)

                # 保存增强后的图片到输出文件夹
                brightened_img.save(output_path)

                print(f"已处理并保存: {output_path}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

print("所有图片处理完成。")
