from sklearn.model_selection import GridSearchCV

# 定义参数范围
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 初始化GridSearch
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='r2')

# 训练GridSearch
grid_search.fit(X_train_scaled, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_}")

# 使用最佳参数重新训练模型
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)

# 评估优化后的模型
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"优化后均方误差 (MSE): {mse_best}")
print(f"优化后决定系数 (R^2): {r2_best}")
