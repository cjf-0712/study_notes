def process_image(pixels, width, height):
    BLOCK_SIZE = 4
    EXTENSION = 1
    TOTAL_SIZE = BLOCK_SIZE + 2 * EXTENSION

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            # 初始化6x6块
            block = [[(0, 0, 0) for _ in range(TOTAL_SIZE)] for _ in range(TOTAL_SIZE)]

            # 填充块数据
            for j in range(TOTAL_SIZE):
                for i in range(TOTAL_SIZE):
                    pixel_x = x + i - EXTENSION
                    pixel_y = y + j - EXTENSION

                    if 0 <= pixel_x < width and 0 <= pixel_y < height:
                        block[j][i] = pixels[pixel_y][pixel_x]
                    else:
                        block[j][i] = (0, 0, 0)

            # 在这里可以对块进行进一步处理
            # 例如：分析块内容、修改块像素等

    return pixels  # 或根据需要返回处理后的数据

# 示例调用
# 假设你有一个图像的像素数据 pixels，宽度 width，高度 height
# processed_pixels = process_image(pixels, width, height)
