# 在加权运算中定义边界处理模式
boundary_mode = 'mirror'  # 可选值: 'constant', 'mirror', 'wrap'

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
            if boundary_mode == 'mirror':
                # 镜像处理
                mirror_xi = xi
                mirror_yj = yj
                if xi < 0:
                    mirror_xi = -xi
                elif xi >= width:
                    mirror_xi = 2 * width - xi - 2
                if yj < 0:
                    mirror_yj = -yj
                elif yj >= height:
                    mirror_yj = 2 * height - yj - 2
                # 确保镜像后的坐标在有效范围内
                mirror_xi = max(0, min(mirror_xi, width - 1))
                mirror_yj = max(0, min(mirror_yj, height - 1))
                row.append(pixels[mirror_xi, mirror_yj])
            elif boundary_mode == 'wrap':
                # 环绕处理
                wrap_xi = xi % width
                wrap_yj = yj % height
                row.append(pixels[wrap_xi, wrap_yj])
            else:
                # 常数填充
                row.append((0, 0, 0))
    block.append(row)
