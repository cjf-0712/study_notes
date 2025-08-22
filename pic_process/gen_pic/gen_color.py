from PIL import Image
import os


def create_custom_image(width, height, save_path):
    """
    创建一张RGB图片，中间一列的G通道为255，其余通道为0，其他像素为黑色。

    :param width: 图片的宽度
    :param height: 图片的高度
    :param save_path: 图片保存的完整路径，包括文件名和扩展名
    """
    # 创建全黑的图像
    image = Image.new("RGB", (width, height), (0, 0, 0))
    pixels = image.load()

    # 计算中间列的索引
    middle_column = height // 2

    # 设置中间列的G通道为255
    for x in range(width):
        pixels[x , middle_column] = (0, 128, 0)

    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    image.save(save_path)
    print(f"图片已成功保存到: {save_path}")


if __name__ == "__main__":
    # 设置图片尺寸
    width = 256  # 您可以根据需要调整宽度
    height = 128  # 您可以根据需要调整高度

    # 设置保存路径
    # 请将以下路径替换为您希望保存图片的实际路径
    # 例如: "C:/Users/您的用户名/Pictures/custom_image.png" 在Windows上
    # 或者 "/Users/您的用户名/Pictures/custom_image.png" 在macOS/Linux上
    save_path = r"F:\FPGA\unilumin_work\pic\col_g128.png"

    create_custom_image(width, height, save_path)
