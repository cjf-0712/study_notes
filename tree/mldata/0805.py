"""
基于随机森林的多因子选股策略
通过L1正则化筛选核心因子 -> 预测图像未来收益
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 模拟数据生成 - 实际应用中应替换为真实数据源
def generate_stock_data(stock_count=500, period_count=60):
    """生成模拟图像数据
    Args:
        stock_count: 图像数量
        period_count: 时间期数（月）
    Returns:
        DataFrame包含因子数据和目标变量
    """
    np.random.seed(42)

    # 创建时间索引和图像代码
    dates = pd.date_range(start='2018-01-01', periods=period_count, freq='M')
    stocks = [f'STK{i:04d}' for i in range(1, stock_count + 1)]

    # 创建多层索引 (日期, 图像代码)
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'ticker'])
    df = pd.DataFrame(index=index)

    # 模拟因子数据（12个候选因子）
    factor_names = [
        'ROE', 'NetProfit_Growth',  # 财务因子
        'MACD', 'Turnover',  # 技术因子
        'PE_Ratio', 'PB_Ratio',  # 估值因子
        'ROA', 'Current_Ratio',  # 财务健康度
        'RSI', 'Bollinger_Width',  # 技术指标
        'Debt_to_Equity', 'EPS_Growth'  # 财务指标
    ]

    for factor in factor_names:
        # 不同因子有不同的分布特征
        if factor in ['ROE', 'ROA']:
            df[factor] = np.random.normal(0.1, 0.05, len(index))
        elif factor in ['NetProfit_Growth', 'EPS_Growth']:
            df[factor] = np.random.normal(0.15, 0.2, len(index))
        elif factor in ['MACD', 'RSI']:
            df[factor] = np.random.normal(0, 1, len(index))
        elif factor == 'Turnover':
            df[factor] = np.random.lognormal(mean=0, sigma=0.5, size=len(index))
        else:
            df[factor] = np.random.rand(len(index)) * 2

    # 添加特征间相关性 (增强数据真实性)
    df['ROE'] = df['ROE'] * 0.7 + df['ROA'] * 0.3
    df['NetProfit_Growth'] = df['NetProfit_Growth'] * 0.8 + df['EPS_Growth'] * 0.2
    df['MACD'] = df['MACD'] * 0.6 - df['RSI'] * 0.4

    # 创建目标变量：下月收益率 (真实情况会有自相关性和因子依赖性)
    # 这里创建真实的因子依赖关系 (后续模型将尝试发现这些关系)
    df['next_month_return'] = (
            0.3 * df['ROE'] +
            0.2 * df['NetProfit_Growth'] -
            0.15 * df['MACD'] +
            0.1 * df['Turnover'] -
            0.05 * df['PE_Ratio'] +
            np.random.normal(0, 0.1, len(index))
    )

    # 添加市场噪声
    market_noise = np.random.randn(len(dates)) * 0.2
    for i, date in enumerate(dates):
        df.loc[date, 'next_month_return'] += market_noise[i]

    return df


# 主程序
def main():
    # ======================== 1. 数据准备 ========================
    print("正在生成模拟图像数据...")
    data = generate_stock_data(stock_count=500, period_count=60)
    print(f"数据集创建完成: {len(data)}条记录，{len(data.columns) - 1}个因子")

    # 定义候选因子和目标变量
    all_factors = [
        'ROE', 'NetProfit_Growth', 'MACD', 'Turnover',
        'PE_Ratio', 'PB_Ratio', 'ROA', 'Current_Ratio',
        'RSI', 'Bollinger_Width', 'Debt_to_Equity', 'EPS_Growth'
    ]
    target = 'next_month_return'

    # 时序数据排序（确保时间连续性）
    data = data.sort_index(level='date')

    # ======================== 2. 因子筛选(L1正则化) ========================
    """
    因子筛选原理:
    使用Lasso回归(L1正则化)筛选核心因子
    L1正则化在损失函数中添加因子系数的绝对值之和作为惩罚项
        L = Σ(y_true - y_pred)² + α * Σ|coef|
    结果：迫使不重要的因子系数趋近于0，保留重要因子
    """
    print("\n正在通过L1正则化筛选核心因子...")

    # 准备数据 (排除最后一个月，因为没有未来收益率)
    X = data[all_factors].values
    y = data[target].values

    # 标准化因子数据 (Lasso对数据尺度敏感)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用时序交叉验证（避免未来信息泄露）
    tscv = TimeSeriesSplit(n_splits=5)

    # 创建LassoCV模型（自动寻找最优alpha）
    lasso = LassoCV(cv=tscv, alphas=np.logspace(-4, 0, 50),
                    max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    # 筛选出系数不为零的因子
    selected_idx = np.where(lasso.coef_ != 0)[0]
    selected_factors = [all_factors[i] for i in selected_idx]

    # 确保只保留8个最重要的因子
    # 按系数绝对值排序取前8
    if len(selected_factors) > 8:
        importance = np.abs(lasso.coef_[selected_idx])
        top8_idx = np.argsort(importance)[-8:]
        selected_factors = [selected_factors[i] for i in top8_idx]

    print(f"\n=== 筛选出的8个核心因子 ===")
    for factor in selected_factors:
        coeff = lasso.coef_[all_factors.index(factor)]
        print(f"{factor}: 系数 = {coeff:.4f}")

    # ======================== 3. 构建随机森林模型 ========================
    """
    随机森林原理:
    1. 构建多棵决策树(Bagging)
    2. 每棵树使用随机样本和随机特征子集
    3. 通过组合多棵树的预测降低方差
    4. 处理高维特征和非线性关系能力强
    """
    print("\n构建随机森林预测模型...")

    # 准备筛选后的因子数据
    X_selected = data[selected_factors].values
    y = data[target].values

    # 标准化筛选后的因子
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected)

    # 划分训练集和测试集（按时间顺序）
    split_idx = int(len(X_selected_scaled) * 0.8)
    X_train = X_selected_scaled[:split_idx]
    y_train = y[:split_idx]
    X_test = X_selected_scaled[split_idx:]
    y_test = y[split_idx:]

    # 创建随机森林模型
    rf = RandomForestRegressor(
        n_estimators=500,  # 树的数量
        max_depth=8,  # 树的最大深度（防止过拟合）
        min_samples_split=5,  # 节点继续分裂的最小样本数
        max_features=0.8,  # 每次分裂考虑的特征比例
        n_jobs=-1,  # 使用所有CPU核心
        random_state=42
    )

    # 训练模型
    rf.fit(X_train, y_train)

    # ======================== 4. 模型评估 ========================
    print("\n评估模型表现...")

    # 预测测试集
    y_pred = rf.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)

    # 计算预测值与实际值相关系数
    correlation = np.corrcoef(y_test, y_pred)[0, 1]

    # 计算预测符号准确率
    sign_match = np.mean(np.sign(y_test) == np.sign(y_pred))

    print(f"测试集性能:")
    print(f"均方误差(MSE): {mse:.6f}")
    print(f"预测与实际值相关系数: {correlation:.4f}")
    print(f"涨跌方向准确率: {sign_match:.2%}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 7))
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.barh(range(len(sorted_idx)), importance[sorted_idx], color='#1f77b4')
    plt.yticks(range(len(sorted_idx)), [selected_factors[i] for i in sorted_idx])
    plt.xlabel("随机森林特征重要性")
    plt.title("核心因子重要性排序")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('factor_importance.png', dpi=300)
    print("因子重要性图已保存为 factor_importance.png")

    # ======================== 5. 图像打分与预测 ========================
    """
    图像打分逻辑:
    1. 使用随机森林预测未来1个月
    2. 预测收益率越高，得分越高
    3. 按排名划分五挡投资组合
    """
    print("\n生成最新图像评分...")

    # 获取最新数据（假设当前时间是数据集中最后一个月）
    current_date = data.index.get_level_values('date').max()
    current_data = data.loc[(current_date, slice(None)), selected_factors]
    current_scaled = scaler_selected.transform(current_data)

    # 预测未来收益
    predicted_returns = rf.predict(current_scaled)

    # 创建评分DataFrame
    scoring_df = pd.DataFrame({
        'ticker': current_data.index.get_level_values('ticker'),
        'predicted_return': predicted_returns
    })

    # 按预测收益率排序
    scoring_df['rank'] = scoring_df['predicted_return'].rank(ascending=False)
    scoring_df['score'] = pd.qcut(scoring_df['rank'], 5,
                                  labels=['E', 'D', 'C', 'B', 'A'])

    # 获取评分最高的前20只图像
    top_stocks = scoring_df[scoring_df['score'] == 'A'].sort_values(
        'predicted_return', ascending=False).head(20)

    # 打印评分分布
    score_counts = scoring_df['score'].value_counts().sort_index()
    print("\n图像评分分布:")
    print(score_counts)

    # 输出建议投资组合
    print("\n=== 推荐投资组合（评分A）===")
    print(top_stocks[['ticker', 'predicted_return']].reset_index(drop=True))


if __name__ == "__main__":
    main()