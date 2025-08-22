import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('alloys.csv')

# 数据预处理
df.fillna(df.mean(), inplace=True)

# 特征与目标变量
X = df[['成分1', '成分2', '成分3', '成分4', '成分5', '润湿性', '硬度']]
y = df['抗拉强度']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = rf.predict(X_test_scaled)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R^2: {r2_score(y_test, y_pred)}")

# 模型优化
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_}")

# 使用最佳模型预测
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
print(f"优化后MSE: {mean_squared_error(y_test, y_pred_best)}")
print(f"优化后R^2: {r2_score(y_test, y_pred_best)}")

# 新数据预测
new_alloy = {
    '成分1': [25],
    '成分2': [30],
    '成分3': [35],
    '成分4': [40],
    '成分5': [45],
    '润湿性': [85],
    '硬度': [210]
}
new_alloy_df = pd.DataFrame(new_alloy)
new_alloy_scaled = scaler.transform(new_alloy_df)
predicted_tensile_strength = best_rf.predict(new_alloy_scaled)
print(f"预测的抗拉强度: {predicted_tensile_strength[0]} MPa")
