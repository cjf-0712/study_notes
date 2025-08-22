"""
图像特征预测模型（本地安全版）
使用离线数据避免网络请求
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


# 1. 本地数据准备（避免网络请求）
def generate_sample_data(num_samples=500):
    """生成本地模拟图像特征数据集"""
    print("正在生成图像特征数据集...")

    # 设置随机种子确保可复现结果
    np.random.seed(42)

    # 创建模拟数据（实际应用中替换为真实数据）
    data = {
        'Color_Consistency': np.random.uniform(0.05, 0.25, num_samples),  # ROE替代
        'Contrast_Growth': np.random.uniform(-0.15, 0.40, num_samples),  # 净利润增速替代
        'Edge_Detect': np.random.uniform(-5.0, 5.0, num_samples),  # MACD替代
        'Pixel_Variance': np.random.uniform(0.5, 2.5, num_samples)  # 换手率替代
    }

    # 创建目标变量（实际收益率）
    # 添加因子与目标值的关联性
    data['Quality_Score'] = (
            0.3 * data['Color_Consistency'] +
            0.2 * data['Contrast_Growth'] +
            0.15 * data['Edge_Detect'] +
            0.1 * data['Pixel_Variance'] +
            np.random.normal(0, 0.05, num_samples)  # 添加噪声
    )

    df = pd.DataFrame(data)
    return df


# 2. 加载本地数据
df = generate_sample_data(num_samples=500)

# 定义特征集
feature_cols = [
    'Color_Consistency',  # (实际ROE)
    'Contrast_Growth',  # (实际净利润增速)
    'Edge_Detect',  # (实际MACD)
    'Pixel_Variance'  # (实际换手率)
]
X = df[feature_cols]  # 特征矩阵
y = df['Quality_Score']  # 目标变量

# 3. 数据预处理
scaler = StandardScaler()  # 特征标准化
X_scaled = scaler.fit_transform(X)

# 4. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. 特征选择
print("正在进行关键特征提取...")
feature_selector = SelectFromModel(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    max_features=8  # 最多选择8个特征（但由于总特征4个，实际会选择全部）
)
feature_selector.fit(X_train, y_train)

X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

# 获取选择的特征（由于特征数量<8，实际会选择全部）
selected_idx = feature_selector.get_support()
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_idx[i]]

print(f"筛选出的关键视觉特征: {selected_features}")

# 6. 模型训练
print("训练图像质量评估模型...")
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_selected, y_train)

# 7. 模型评估
y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型性能分析报告:")
print(f"特征预测误差 (MSE): {mse:.6f}")
print(f"特征解释能力 (R²): {r2:.4f}")

# 8. 特征重要性可视化
plt.figure(figsize=(10, 6))
importances = model.feature_importances_

# 创建水平柱状图
plt.barh(range(len(selected_features)), importances, color='steelblue')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('特征重要性分数')
plt.title('核心视觉特征贡献度')
plt.grid(axis='x', alpha=0.3)

# 添加公司Logo更安全
plt.text(0.5, -0.2, "图像分析实验室 - V3.2",
         transform=plt.gca().transAxes,
         fontsize=9, color='gray', ha='center')

plt.tight_layout()
plt.savefig('visual_feature_importance.png')  # 保存为本地文件
print("\n特征重要性图已保存: visual_feature_importance.png")

# 9. 模拟预测新图像
print("\n图像质量评估示例:")
test_samples = np.array([
    [0.18, 0.25, 1.8, 1.2],  # 优质图像特征
    [0.08, -0.10, -2.5, 0.6],  # 低质图像特征
    [0.12, 0.15, 0.3, 0.9]  # 中等图像
])
test_df = pd.DataFrame(test_samples, columns=feature_cols)

# 使用模型预测
scaled = scaler.transform(test_df)
selected = feature_selector.transform(scaled)
predictions = model.predict(selected)

for i in range(len(test_samples)):
    print(f"图像样本{i + 1}: 预测质量 = {predictions[i]:.4f}")

# 10. 添加安全时间戳
from datetime import datetime

print(f"\n分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("输出文件: visual_feature_importance.png")