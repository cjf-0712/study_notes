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

# 示例像素块（假设每个像素有三个通道：R, G, B）
block = [
    [(255, 0, 0), (200, 50, 50)],
    [(150, 100, 100), (100, 150, 150)]
]

# 定义像素组（例如左上、右上、左下、右下四个像素）
group1 = [block[0][0], block[0][1], block[1][0], block[1][1]]

# 定义对应的权重
weights = [0.1, 0.2, 0.3, 0.4]

# 计算各个通道的加权和
weighted_r = weighted_sum_channel(group1, 0, weights)
weighted_g = weighted_sum_channel(group1, 1, weights)
weighted_b = weighted_sum_channel(group1, 2, weights)

print(f"加权后的 R 通道值: {weighted_r}")
print(f"加权后的 G 通道值: {weighted_g}")
print(f"加权后的 B 通道值: {weighted_b}")
