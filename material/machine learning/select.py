# 读取CSV文件
df = pd.read_csv('alloys.csv')

# 或者读取Excel文件

# df = pd.read_excel('alloys.xlsx')
import pandas as pd

# 读取CSV文件
df = pd.read_csv('alloys.csv')

# 设置筛选条件
wettability_threshold = 85
hardness_threshold = 220
tensile_strength_threshold = 420

# 筛选合金
filtered_df = df[
    (df['润湿性'] >= wettability_threshold) &
    (df['硬度'] <= hardness_threshold) &
    (df['抗拉强度'] >= tensile_strength_threshold)
]

# 输出结果
print("符合条件的合金有：")
print(filtered_df[['合金名称', '润湿性', '硬度', '抗拉强度']])

import pandas as pd

# 示例数据
data = {
    '合金名称': ['合金A', '合金B', '合金C', '合金D', '合金E'],
    '成分1': [10, 20, 30, 40, 50],
    '成分2': [15, 25, 35, 45, 55],
    '成分3': [20, 30, 40, 50, 60],
    '成分4': [25, 35, 45, 55, 65],
    '成分5': [30, 40, 50, 60, 70],
    '润湿性': [80, 75, 90, 85, 95],          # 范围0-100
    '硬度': [200, 250, 220, 210, 230],        # HV值
    '抗拉强度': [400, 450, 420, 410, 430]     # MPa
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置筛选条件
# 例如：润湿性 >=85，硬度 <=220，抗拉强度 >=420
wettability_threshold = 85
hardness_threshold = 220
tensile_strength_threshold = 420

# 筛选合金
filtered_df = df[
    (df['润湿性'] >= wettability_threshold) &
    (df['硬度'] <= hardness_threshold) &
    (df['抗拉强度'] >= tensile_strength_threshold)
]

# 输出结果
print("符合条件的合金有：")
print(filtered_df[['合金名称', '润湿性', '硬度', '抗拉强度']])
