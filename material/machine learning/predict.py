# 示例新合金数据
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

# 数据标准化
new_alloy_scaled = scaler.transform(new_alloy_df)

# 预测抗拉强度
predicted_tensile_strength = best_rf.predict(new_alloy_scaled)
print(f"预测的抗拉强度: {predicted_tensile_strength[0]} MPa")
