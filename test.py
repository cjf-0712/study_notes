import os
from PIL import Image

# 定义输入和输出目录
input_dir = r'F:\FPGA\unilumin_work\pic\1'
output_dir = r'F:\FPGA\unilumin_work\processed_pic\2'

# 如果输出目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像
        with Image.open(input_path) as img:
            img = img.convert('RGB')  # 确保图像是RGB模式
            pixels = img.load()
            width, height = img.size

            for y in range(height):
                for x in range(width):
                    # 计算当前像素在4x4单元中的位置
                    row_in_unit = y % 4
                    col_in_unit = x % 4

                    r, g, b = pixels[x, y]  # 获取当前像素的RGB值

                    if row_in_unit == 0:
                        # 第一行
                        if col_in_unit == 0:
                            # 移除第一列像素
                            pixels[x, y] = (0, 0, 0)
                        elif col_in_unit == 1:
                            # 保留红色像素
                            pixels[x, y] = (r, 0, 0)
                        elif col_in_unit == 2:
                            # 保留绿色像素
                            pixels[x, y] = (0, g, 0)
                        elif col_in_unit == 3:
                            # 保留蓝色像素
                            pixels[x, y] = (0, 0, b)
                    elif row_in_unit == 1:
                        # 第二行
                        if col_in_unit == 0:
                            # 保留绿色像素
                            pixels[x, y] = (0, g, 0)
                        elif col_in_unit == 1:
                            # 保留蓝色像素
                            pixels[x, y] = (0, 0, b)
                        elif col_in_unit == 2:
                            # 移除第三列像素
                            pixels[x, y] = (0, 0, 0)
                        elif col_in_unit == 3:
                            # 保留红色像素
                            pixels[x, y] = (r, 0, 0)
                    elif row_in_unit == 2:
                        # 第三行
                        if   col_in_unit == 0:
                            # 保留红色像素
                            pixels[x, y] = (r, 0, 0)
                        elif col_in_unit == 1:
                            # 移除第二列像素
                            pixels[x, y] = (0, 0, 0)
                        elif col_in_unit == 2:
                            # 保留蓝色像素
                            pixels[x, y] = (0, 0, b)
                        elif col_in_unit == 3:
                            # 保留绿色像素
                            pixels[x, y] = (0, g, 0)
                    elif row_in_unit == 3:
                        # 第四行
                        if col_in_unit == 0:
                            # 保留蓝色像素
                            pixels[x, y] = (0, 0, b)
                        elif col_in_unit == 1:
                            # 保留绿色像素
                            pixels[x, y] = (0, g, 0)
                        elif col_in_unit == 2:
                            # 保留红色像素
                            pixels[x, y] = (r, 0, 0)
                        elif col_in_unit == 3:
                            # 移除第四列像素
                            pixels[x, y] = (0, 0, 0)

                    # 更新像素值
                    #pixels[x, y] = (r, g, b)

            # 保存处理后的图像
            img.save(output_path)

        print(f"已处理并保存: {output_path}")

print("所有图像处理完成！")
