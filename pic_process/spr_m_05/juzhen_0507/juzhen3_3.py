for y in range(0, height, 4):
    for x in range(0, width, 4):
        # 定义当前4x4块的像素数据基础上，再额外增加6*6的矩阵数据
        # block = [[pixels[x + i, y + j] for i in range(-1, 5)] for j in range(-1, 5)]
        block = []
        for j in range(-1, 5):
            row = []
            for i in range(-1, 5):
                xi = x + i
                yj = y + j
                # 边界检查
                if 0 <= xi < width and 0 <= yj < height:
                    row.append(pixels[xi, yj])
                else:
                    # 可以选择填充黑色或其他默认值
                    row.append((0, 0, 0))
            block.append(row)


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


        # 检查边界条件
        # 是否在左边界
        lef_b = (x == 0 or x == 1)
        # 是否在右边界
        rig_b = (x == width - 1 or x == width - 2)
        # 是否在上边界
        top_b = (y == 0 or y == 1)
        # 是否在下边界末行
        bot_b = (y == height - 1 or height - 2)

        # 处理 (0,0): 变为黑色
        pixels[x + 0, y + 0] = (0, 0, 0)

        # 灰色权重
        weight1 = [0.22, 0.32, 0.45, 0.26, 0.37, 1, 0.45, 0.22, 0.37, 0.32]
        # 黄色权重
        weight2 = [0.32, 0.37, 0.22, 0.45, 1, 0.37, 0.26, 0.37, 0.32, 0.22]
        # 蓝色权重
        weight3 = [0.37, 0.32, 0.37, 1, 0.45, 0.22, 0.32, 0.45, 0.26, 0.22]
        # 橙色权重
        weight4 = [0.22, 0.26, 0.45, 0.32, 0.22, 0.45, 1, 0.37, 0.32, 0.37]
        # 处理 (0,1): G数据
        # 边界情况的算式
        group1 = [block[0][-1], block[-1][0], block[0][0], block[1][0], block[-1][1],
                  block[0][1], block[1][1], block[2][1], block[0][2], block[1][2]]

        g_avg1 = weighted_sum_channel(group1, 1, weight1)


        # 处理 (1,0): R数据
        group2 = [block[0][-1], block[1][-1], block[-1][0], block[0][0], block[1][0],
                  block[2][0], block[0][1], block[1][1], block[2][1], block[1][2]]

        r_avg = weighted_sum_channel(group2, 0, weight2)
        pixels[x + 1, y + 0] = (r_avg, 0, 0)

        # 处理 (1,1): B数据
        group3 = [block[1][-1], block[0][0], block[1][0], block[2][0], block[-1][1],
                  block[0][1], block[1][1], block[2][1], block[0][2], block[1][2]]

        b_avg = weighted_sum_channel(group3, 2, weight4)
        pixels[x + 1, y + 1] = (0, 0, b_avg)

        # 处理 (1,2): 变为黑色
        pixels[x + 1, y + 2] = (0, 0, 0)

        # 处理 (0,2): R数据
        group4 = [block[0][0], block[1][0], block[-1][1], block[0][1], block[1][1],
                  block[2][1], block[-1][2], block[0][2], block[1][2], block[0][3]]
        r_avg2 = weighted_sum_channel(group4, 0, weight3)
        pixels[x + 0, y + 2] = (r_avg2, 0, 0)

        # 处理 (0,3): B数据
        group5 = [block[0][1], block[-1][2], block[0][2], block[1][2], block[-1][3],
                  block[0][3], block[1][3], block[2][3], block[0][4], block[1][4]]
        b_avg2 = weighted_sum_channel(group5, 2, weight1)
        pixels[x + 0, y + 3] = (0, 0, b_avg2)

        # 处理 (1,3): G数据
        group6 = [block[1][1], block[0][2], block[1][2], block[2][2], block[-1][3],
                  block[0][3], block[1][3], block[2][3], block[0][4], block[1][4]]
        g_avg2 = weighted_sum_channel(group6, 1, weight4)
        pixels[x + 1, y + 3] = (0, g_avg2, 0)

        # 处理 (2,1): 变为黑色
        pixels[x + 2, y + 1] = (0, 0, 0)

        # 处理 (3,1): R数据
        group7 = [block[3][-1], block[2][0], block[3][0], block[4][0], block[1][1],
                  block[2][1], block[3][1], block[4][1], block[2][2], block[3][2]]
        r_avg3 = weighted_sum_channel(group7, 0, weight4)
        pixels[x + 3, y + 1] = (r_avg3, 0, 0)

        # 处理 (3,0): b数据
        group8 = [block[2][-1], block[3][-1], block[1][0], block[2][0], block[3][0],
                  block[4][0], block[2][1], block[3][1], block[4][1], block[3][2]]
        b_avg3 = weighted_sum_channel(group8, 2, weight2)
        pixels[x + 3, y + 0] = (0, 0, b_avg3)

        # 处理 (2,0): G数据
        group9 = [block[2][-1], block[3][-1], block[1][0], block[2][0], block[3][0],
                  block[4][0], block[1][1], block[2][1], block[3][1], block[2][2]]
        g_avg3 = weighted_sum_channel(group9, 1, weight3)
        pixels[x + 2, y + 0] = (0, g_avg3, 0)

        # 处理 (3,3): 变为黑色
        pixels[x + 3, y + 3] = (0, 0, 0)

        # 处理 (2,2): B数据
        group10 = [block[2][0], block[3][0], block[1][1], block[2][1], block[3][1],
                   block[4][1], block[1][2], block[2][2], block[3][2], block[2][3]]
        b_avg4 = weighted_sum_channel(group10, 2, weight3)
        pixels[x + 2, y + 2] = (0, 0, b_avg4)

        # 处理 (3,2): G数据
        group11 = [block[2][0], block[3][0], block[1][1], block[2][1], block[3][1],
                   block[4][1], block[2][2], block[3][2], block[4][2], block[3][3]]
        g_avg4 = weighted_sum_channel(group11, 1, weight2)
        pixels[x + 3, y + 2] = (0, g_avg4, 0)