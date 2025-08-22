from PIL import Image
import os


def process_image(input_path, output_path):
    """处理单个图像，实现智能补色卷积"""
    with Image.open(input_path) as img:
        rgb_img = img.convert('RGB')
        width, height = rgb_img.size

        # 创建新图像并初始化边缘像素
        new_img = Image.new('RGB', (width, height))
        orig_pixels = rgb_img.load()
        new_pixels = new_img.load()

        # 预处理：直接复制边缘像素
        for y in [0, height - 1]:
            for x in range(width):
                new_pixels[x, y] = orig_pixels[x, y]
        for x in [0, width - 1]:
            for y in range(1, height - 1):
                new_pixels[x, y] = orig_pixels[x, y]

        # 核心处理逻辑
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                sum_r = sum_g = sum_b = 0
                valid_neighbors = 0

                # 四邻域偏移量：上、下、左、右
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    px = x + dx
                    py = y + dy

                    # 安全边界检查
                    if 0 <= px < width and 0 <= py < height:
                        r, g, b = orig_pixels[px, py]
                        sum_r += r
                        sum_g += g
                        sum_b += b
                        valid_neighbors += 1

                # 计算有效平均值（防止除以零）
                if valid_neighbors > 0:
                    avg_r = sum_r // valid_neighbors
                    avg_g = sum_g // valid_neighbors
                    avg_b = sum_b // valid_neighbors
                else:
                    avg_r = avg_g = avg_b = 0

                # 获取当前像素值
                curr_r, curr_g, curr_b = orig_pixels[x, y]

                # 智能补色算法
                new_r = curr_r + max(0, (avg_r - curr_r) // 2)  # 补差值的一半
                new_g = curr_g + max(0, (avg_g - curr_g) // 2)
                new_b = curr_b + max(0, (avg_b - curr_b) // 2)

                # 应用处理条件
                new_r = min(255, new_r) if curr_r < avg_r else curr_r
                new_g = min(255, new_g) if curr_g < avg_g else curr_g
                new_b = min(255, new_b) if curr_b < avg_b else curr_b

                new_pixels[x, y] = (new_r, new_g, new_b)

        new_img.save(output_path)


def batch_process(input_folder, output_folder):
    """批量处理文件夹中的图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                process_image(input_path, output_path)
                print(f"成功处理：{filename}")
            except Exception as e:
                print(f"处理失败：{filename} - {str(e)}")


if __name__ == "__main__":
    input_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\test_160"
    output_folder = r"F:\FPGA\unilumin_work\pic\pic_ku\final\buse_0507_3"
    batch_process(input_folder, output_folder)
