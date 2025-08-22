"""
基于随机森林的多因子选股策略（增强版）
- 完整处理数据质量问题（NaN/无穷大/极值）
- 包含详细日志和可视化
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quant_strategy.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# 数据质量检查和处理工具函数
def clean_data(X, y, feature_names, target_name):
    """清洗数据：处理NaN、Inf和极端值"""
    logger.info("开始数据清洗...")

    # 处理目标变量
    if y is not None:
        # 处理Inf
        if np.any(np.isinf(y)):
            y_median = np.median(y[~np.isinf(y)])
            y = np.where(np.isinf(y), y_median, y)
            logger.info(f"目标变量{target_name}中的Inf值已用中位数({y_median:.4f})替换")

        # 处理NaN
        if np.any(np.isnan(y)):
            y_median = np.nanmedian(y)
            y = np.where(np.isnan(y), y_median, y)
            logger.info(f"目标变量{target_name}中的NaN值已用中位数({y_median:.4f})替换")

        # 处理极端值
        if np.any(np.abs(y) > 1e4):
            q1 = np.percentile(y, 1)
            q99 = np.percentile(y, 99)
            y = np.clip(y, q1, q99)
            logger.info(f"目标变量{target_name}极端值已缩尾处理到[{q1:.4f}, {q99:.4f}]")

    # 处理特征矩阵
    for i in range(X.shape[1]):
        col = X[:, i]
        factor = feature_names[i] if feature_names else f"Feature_{i}"

        # 处理Inf
        if np.any(np.isinf(col)):
            col_median = np.median(col[~np.isinf(col)])
            col = np.where(np.isinf(col), col_median, col)
            logger.info(f"因子{factor}中的Inf值已用中位数({col_median:.4f})替换")

        # 处理NaN
        if np.any(np.isnan(col)):
            col_median = np.nanmedian(col)
            col = np.where(np.isnan(col), col_median, col)
            logger.info(f"因子{factor}中的NaN值已用中位数({col_median:.4f})替换")

        # 处理极端值
        if np.any(np.abs(col) > 1e4):
            q1 = np.percentile(col, 1)
            q99 = np.percentile(col, 99)
            col = np.clip(col, q1, q99)
            logger.info(f"因子{factor}极端值已缩尾处理到[{q1:.4f}, {q99:.4f}]")

        X[:, i] = col

    # 最终数据质量报告
    if y is not None:
        y_nan_count = np.isnan(y).sum()
        y_inf_count = np.isinf(y).sum()
        logger.info(f"目标变量清洗后: NaN={y_nan_count}, Inf={y_inf_count}")

    X_nan_count = np.isnan(X).sum()
    X_inf_count = np.isinf(X).sum()
    logger.info(f"特征矩阵清洗后: 总NaN={X_nan_count}, 总Inf={X_inf_count}")

    logger.info("数据清洗完成!")
    return X, y


# 模拟数据生成（稳定版）
def generate_stock_data(stock_count=500, period_count=60):
    """生成模拟图像数据（加强稳定性）"""
    logger.info(f"生成模拟数据: {stock_count}只图像 x {period_count}个月")
    np.random.seed(42)

    # 创建时间索引和图像代码
    dates = pd.date_range(start='2018-01-01', periods=period_count, freq='M')
    stocks = [f'STK{i:04d}' for i in range(1, stock_count + 1)]

    # 创建多层索引 (日期, 图像代码)
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'ticker'])
    df = pd.DataFrame(index=index)

    # 因子列表（12个候选因子）
    factor_names = [
        'ROE', 'NetProfit_Growth',  # 财务因子
        'MACD', 'Turnover',  # 技术因子
        'PE_Ratio', 'PB_Ratio',  # 估值因子
        'ROA', 'Current_Ratio',  # 财务健康度
        'RSI', 'Bollinger_Width',  # 技术指标
        'Debt_to_Equity', 'EPS_Growth'  # 财务指标
    ]

    # 基础因子值生成（避免零除错误）
    for factor in factor_names:
        if factor in ['ROE', 'ROA']:
            # 添加小常数避免零值
            df[factor] = np.abs(np.random.normal(0.1, 0.05, len(index))) + 0.001
        elif factor in ['NetProfit_Growth', 'EPS_Growth']:
            df[factor] = np.random.normal(0.15, 0.2, len(index)) + 0.001
        elif factor in ['MACD', 'RSI']:
            df[factor] = np.random.normal(0, 1, len(index))
        elif factor == 'Turnover':
            # 使用指数分布避免过大值
            df[factor] = np.random.exponential(1.0, len(index))
        else:
            # 有限范围内的值
            df[factor] = np.random.rand(len(index)) * 2

    # 添加特征间相关性 (增强数据真实性)
    df['ROE'] = df['ROE'] * 0.7 + df['ROA'] * 0.3
    df['NetProfit_Growth'] = df['NetProfit_Growth'] * 0.8 + df['EPS_Growth'] * 0.2
    df['MACD'] = df['MACD'] * 0.6 - df['RSI'] * 0.4

    # 创建目标变量（确保范围合理）
    df['next_month_return'] = (
            0.3 * df['ROE'] +
            0.2 * df['NetProfit_Growth'] -
            0.15 * df['MACD'] +
            0.1 * df['Turnover'] -
            0.05 * df['PE_Ratio'] +
            np.random.normal(0, 0.1, len(index))
    )

    # 添加市场噪声（有界噪声）
    market_noise = np.random.normal(0, 0.1, len(dates))
    market_noise = np.clip(market_noise, -0.5, 0.5)  # 限制幅度
    for i, date in enumerate(dates):
        df.loc[date, 'next_month_return'] += market_noise[i]

    # 目标变量缩尾处理
    q1 = df['next_month_return'].quantile(0.01)
    q99 = df['next_month_return'].quantile(0.99)
    df['next_month_return'] = np.clip(df['next_month_return'], q1, q99)

    logger.info(f"数据生成完成! 目标变量范围: [{q1:.4f}, {q99:.4f}]")
    return df


# 主策略函数
def run_quant_strategy():
    start_time = time.time()
    logger.info("=== 多因子图像优化策略启动 ===")

    # ========== 1. 数据准备 ==========
    try:
        data = generate_stock_data(stock_count=500, period_count=60)
        logger.info(f"数据准备完成: {len(data)}条记录")
    except Exception as e:
        logger.error(f"数据生成失败: {str(e)}")
        return

    # 定义候选因子和目标变量
    all_factors = [
        'ROE', 'NetProfit_Growth', 'MACD', 'Turnover',
        'PE_Ratio', 'PB_Ratio', 'ROA', 'Current_Ratio',
        'RSI', 'Bollinger_Width', 'Debt_to_Equity', 'EPS_Growth'
    ]
    target = 'next_month_return'

    # 时序数据排序
    data = data.sort_index(level='date')

    # 分离特征和目标
    X = data[all_factors].values
    y = data[target].values

    # 数据清洗
    X_clean, y_clean = clean_data(X, y, all_factors, target)

    # ========== 2. 因子筛选(L1正则化) ==========
    logger.info("开始因子筛选(L1正则化)...")

    # 标准化因子数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 使用时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    # 创建LassoCV模型
    try:
        lasso = LassoCV(cv=tscv, alphas=np.logspace(-4, 0, 50),
                        max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y_clean)
        logger.info("L1正则化因子筛选完成!")
    except Exception as e:
        logger.error(f"LassoCV拟合失败: {str(e)}")
        return

    # 筛选出系数不为零的因子
    selected_idx = np.where(lasso.coef_ != 0)[0]
    selected_factors = [all_factors[i] for i in selected_idx]

    # 确保只保留8个最重要的因子
    if len(selected_factors) > 8:
        importance = np.abs(lasso.coef_[selected_idx])
        top8_idx = np.argsort(importance)[-8:]
        selected_factors = [selected_factors[i] for i in top8_idx]

    # 记录筛选结果
    logger.info(f"筛选出的{len(selected_factors)}个核心因子:")
    for factor in selected_factors:
        coeff = lasso.coef_[all_factors.index(factor)]
        logger.info(f"  {factor}: 系数 = {coeff:.4f}")

    # ========== 3. 构建随机森林模型 ==========
    logger.info("构建随机森林模型...")

    # 准备筛选后的因子数据
    X_selected = data[selected_factors].values
    y = y_clean  # 使用清洗后的目标值

    # 标准化筛选后的因子
    scaler_selected = StandardScaler()
    X_selected_scaled = scaler_selected.fit_transform(X_selected)

    # 划分训练集和测试集（按时间顺序）
    split_idx = int(len(X_selected_scaled) * 0.8)
    X_train = X_selected_scaled[:split_idx]
    y_train = y[:split_idx]
    X_test = X_selected_scaled[split_idx:]
    y_test = y[split_idx:]

    logger.info(f"数据集划分: 训练集={len(X_train)}, 测试集={len(X_test)}")

    # 创建随机森林模型
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )

    # 训练模型
    try:
        rf.fit(X_train, y_train)
        logger.info("随机森林训练完成!")
    except Exception as e:
        logger.error(f"随机森林训练失败: {str(e)}")
        return

    # ========== 4. 模型评估 ==========
    logger.info("评估模型性能...")

    # 预测测试集
    y_pred = rf.predict(X_test)

    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    sign_match = np.mean(np.sign(y_test) == np.sign(y_pred))

    # 记录性能
    logger.info(f"测试集性能:")
    logger.info(f"  均方误差(MSE): {mse:.6f}")
    logger.info(f"  预测与实际值相关系数: {correlation:.4f}")
    logger.info(f"  涨跌方向准确率: {sign_match:.2%}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 7))
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.barh(range(len(sorted_idx)), importance[sorted_idx], color='#1f77b4')
    plt.yticks(range(len(sorted_idx)), [selected_factors[i] for i in sorted_idx])
    plt.xlabel("特征重要性", fontsize=12)
    plt.title("核心因子重要性排序", fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('factor_importance.png', dpi=300)
    logger.info("因子重要性图已保存为 factor_importance.png")

    # ========== 5. 图像打分与预测 ==========
    logger.info("生成最新图像评分...")

    # 获取最新数据（数据集中最后一个月）
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
    scoring_df = scoring_df.sort_values('predicted_return', ascending=False)
    scoring_df['rank'] = scoring_df['predicted_return'].rank(ascending=False)
    scoring_df['score'] = pd.qcut(scoring_df['rank'], 5,
                                  labels=['E', 'D', 'C', 'B', 'A'])

    # 获取评分最高的前20只图像
    top_stocks = scoring_df[scoring_df['score'] == 'A' \
                                                   ''].head(20)

    # 保存评分结果
    scoring_df.to_csv('stock_scores.csv', index=False)
    logger.info(f"图像评分已保存至 stock_scores.csv")

    # 输出简要结果
    logger.info(f"最佳20只图像预测收益:")
    for i, (idx, row) in enumerate(top_stocks.iterrows()):
        logger.info(f"{i + 1}. {row['ticker']}: 预测收益={row['predicted_return']:.4f}")

    # 策略总结
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"策略完成! 总耗时: {runtime:.2f}秒")

    return scoring_df


if __name__ == "__main__":
    logger.info("程序启动")
    run_quant_strategy()
    logger.info("程序结束")