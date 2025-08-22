
import os
from PIL import Image

def process_image(image_path, output_path):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = img.load()
        width, height = img.size
        # img.size是（w,h）的元组，相当于把元组中的数据赋值给width和height

        # 确保图片的宽度和高度是4的倍数
        if width % 4 != 0 or height % 4 != 0:
            print(f"图片 {image_path} 的尺寸不是4的倍数，跳过处理。")
            return

        for y in range(0, height, 6):
            for x in range(0, width, 6):
                # 定义当前6x6块的像素数据
                block = [[pixels[x + i, y + j] for i in range(6)] for j in range(6)]

                # 定义函数来计算平均值，channel0，1，2分别代表rgb
                def weighted_sum_channel(group, channel, weights):
                    """
                    计算指定通道的加权和。

                    :param group: 包含像素的列表，每个像素是一个可索引的序列（如元组或列表）。
                    :param channel: 指定的通道索引。
                    :param weights: 对应每个像素的权重列表。
                    :return: 加权和的整数值。
                    """
                    if len(group) != len(weights):
                        raise ValueError("像素组和权重列表的长度必须相同。")

                    weighted_sum = sum(pixel[channel] * weight for pixel, weight in zip(group, weights))
                    return int(weighted_sum)
                #        weighted_sum_channel(group1, 0, weights)
                weight_r = [0.63, 0.63, 0.63, 0.63]
                weight_g = [0.65, 0.65, 0.65, 0.65]
                weight_b = [0.7, 0.7, 0.7, 0.7]
                group1 = [block[0][0], block[0][1], block[1][0], block[1][1]]
                # 处理 (0,0): 变为黑色
                pixels[x + 0, y + 0] = (0, 0, 0)

                # 处理 (0,1): G数据
                g_avg = weighted_sum_channel(group1, 1, weight_g)
                pixels[x + 0, y + 1] = (0, g_avg, 0)

                # 处理 (1,0): R数据

                r_avg = weighted_sum_channel(group1, 0, weight_r)
                pixels[x + 1, y + 0] = (r_avg, 0, 0)

                # 处理 (1,1): B数据
                b_avg = weighted_sum_channel(group1, 2, weight_b)
                pixels[x + 1, y + 1] = (0, 0, b_avg)

                # 处理 (1,2): 变为黑色
                pixels[x + 1, y + 2] = (0, 0, 0)

                # 处理 (1,3): G数据
                group2 = [block[0][2], block[0][3], block[1][2], block[1][3]]
                g_avg2 = weighted_sum_channel(group2, 1, weight_g)
                pixels[x + 1, y + 3] = (0, g_avg2, 0)

                # 处理 (0,3): B数据
                b_avg2 = weighted_sum_channel(group2, 2, weight_b)
                pixels[x + 0, y + 3] = (0, 0, b_avg2)

                # 处理 (0,2): R数据
                r_avg2 = weighted_sum_channel(group2, 0, weight_r)
                pixels[x + 0, y + 2] = (r_avg2, 0, 0)

                # 处理 (2,1): 变为黑色
                pixels[x + 2, y + 1] = (0, 0, 0)

                # 处理 (3,1): R数据
                group3 = [block[2][0], block[2][1], block[3][0], block[3][1]]
                r_avg3 = weighted_sum_channel(group3, 0, weight_r)
                pixels[x + 3, y + 1] = (r_avg3, 0, 0)

                # 处理 (3,0): b数据
                b_avg3 = weighted_sum_channel(group3, 2, weight_b)
                pixels[x + 3, y + 0] = (0, 0, b_avg3)

                # 处理 (2,0): G数据
                g_avg3 = weighted_sum_channel(group3, 1, weight_g)
                pixels[x + 2, y + 0] = (0, g_avg3, 0)

                # 处理 (3,3): 变为黑色
                pixels[x + 3, y + 3] = (0, 0, 0)

                # 处理 (2,2): B数据
                group4 = [block[2][2], block[2][3], block[3][2], block[3][3]]
                b_avg4 = weighted_sum_channel(group4, 2, weight_b)
                pixels[x + 2, y + 2] = (0, 0, b_avg4)

                # 处理 (3,2): G数据
                g_avg4 = weighted_sum_channel(group4, 1, weight_g)
                pixels[x + 3, y + 2] = (0, g_avg4, 0)

                # 处理 (2,3): R数据
                r_avg4 = weighted_sum_channel(group4, 0, weight_r)
                pixels[x + 2, y + 3] = (r_avg4, 0, 0)

        # 保存处理后的图片到输出路径
        img.save(output_path)
        print(f"已保存处理后的图片到 {output_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")


def process_all_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"正在处理 {input_path}...")
            process_image(input_path, output_path)


if __name__ == "__main__":
    input_directory = r"F:\FPGA\unilumin_work\pic\caitu"
    output_directory = r"F:\FPGA\unilumin_work\pic\weightsum_TEST6"

    process_all_images(input_directory, output_directory)
    print("所有图片处理完成。")
