def calculate_adjusted_expectation(
        O: float,  # 帧开始点帧率（基准帧率）
        S1: float,  # 第一阶段掉帧点（绝对帧率）
        T1: float,  # 第一阶段加帧点（绝对帧率）
        M: float,  # 调整触发点（绝对帧率）
        S2: float,  # 第二阶段掉帧点（绝对帧率）
        T2: float,  # 第二阶段加帧点（绝对帧率）
        p1: float,  # 第一阶段触发概率
        p2: float  # 第二阶段成功概率
) -> dict:
    """
    计算帧率调整后的期望帧值和关键指标
    假设初始帧率为O（通常S1 < O < T1，M位于O和T1之间）
    """
    # 1. 验证参数有效性
    if not (min(S1, S2) <= M <= max(T1, T2)):
        return {"error": "调整触发点M必须在掉帧和加帧区间内"}

    if not (0 < p1 <= 1) or not (0 < p2 <= 1):
        return {"error": "概率必须在0到1之间"}

    # 2. 计算关键指标（使用绝对帧率计算点差）
    # 第一阶段指标
    stage1_risk = abs(O - S1)  # 第一阶段抖动帧差
    stage1_reward = abs(T1 - O)  # 第一阶段潜在增益

    # 第二阶段指标
    stage2_risk = abs(M - S2)  # 第二阶段抖动帧差
    stage2_reward = abs(T2 - M)  # 第二阶段潜在增益

    # 3. 计算调整后的抖动收益比
    risk_reward_ratio = stage2_risk / stage2_reward if stage2_reward else float('inf')

    # 4. 计算整体期望帧值（以帧开始点O为基准）
    # 路径1: 第一阶段失败（掉帧到S1）
    path1_gain = S1 - O  # 帧率下降，负值

    # 路径2: 第一阶段成功但第二阶段失败
    path2_gain = S2 - O  # 帧率下降，负值

    # 路径3: 第一阶段成功且第二阶段成功
    path3_gain = T2 - O  # 帧率提升，正值

    # 期望帧值计算
    expectation = (
            (1 - p1) * path1_gain +
            p1 * (1 - p2) * path2_gain +
            p1 * p2 * path3_gain
    )

    # 5. 计算清晰平衡成功率（第二阶段要求）
    required_win_rate = stage2_risk / (stage2_risk + stage2_reward) if (stage2_risk + stage2_reward) else 0

    # 6. 分析调整效果
    adjustment_effect = ""
    initial_risk = stage1_risk

    # 比较调整前后抖动帧差
    adjusted_risk = stage2_risk
    if adjusted_risk < initial_risk:  # 抖动降低
        adjustment_effect = "掉帧容忍度缩窄"
        tolerance = f"可容忍成功率降低至: {required_win_rate:.2%}"
    else:  # 抖动增加
        adjustment_effect = "掉帧容忍度放宽"
        tolerance = f"需要补偿成功率提高至: {required_win_rate:.2%}"

    # 7. 返回完整结果
    return {
        # 核心结果
        "overall_expectation": expectation,
        "expected_frame_gain": expectation,  # 期望帧帧率增益
        "overall_success_rate": p1 * p2,
        "required_success_rate_stage2": required_win_rate,
        "actual_success_rate_stage2": p2,
        "is_positive_expectation": expectation > 0,
        "adjustment_effect": adjustment_effect,
        "risk_tolerance_analysis": tolerance,

        # 详细指标
        "risk_reward_ratio": risk_reward_ratio,
        "stage1_success_rate": p1,
        "M_position_ratio": (M - min(S1, S2)) / (max(T1, T2) - min(S1, S2)) if max(T1, T2) != min(S1, S2) else 0,
        "stage2_risk": stage2_risk,
        "stage2_reward": stage2_reward,
        "recommendation": (
            "建议调整参数" if (p2 < required_win_rate)
            else "满足性能提升要求"
        )
    }


# -------------------------------
# 示例测试（三种典型场景）
# -------------------------------
if __name__ == "__main__":
    # 场景1：掉帧容忍度缩窄（安全边际提升）
    print("\n场景1：掉帧容忍度缩窄（180→30）")
    result1 = calculate_adjusted_expectation(
        O=8100,  # 初始帧率
        S1=8000, T1=8300, M=8150,  # 第一阶段参数
        S2=8120, T2=8280,  # 调整后参数
        p1=0.65, p2=0.60  # 成功率
    )
    if "error" in result1:
        print(f"错误: {result1['error']}")
    else:
        for k, v in result1.items():
            print(f"{k:>30}: {v}")

    # 场景2：掉帧容忍度放宽（抖动增加）
    print("\n场景2：掉帧容忍度放宽（180→230）")
    result2 = calculate_adjusted_expectation(
        O=8100,
        S1=8000, T1=8300, M=8150,
        S2=7970, T2=8250,  # 放宽掉帧容忍度
        p1=0.65, p2=0.78  # 需要更高成功率补偿
    )
    if "error" in result2:
        print(f"错误: {result2['error']}")
    else:
        for k, v in result2.items():
            print(f"{k:>30}: {v}")

    # 场景3：M点接近加帧点（高抖动位置）
    print("\n场景3：M点接近加帧点（高波动位置）")
    result3 = calculate_adjusted_expectation(
        O=8100,
        S1=8000, T1=8300, M=8280,  # M点接近加帧目标
        S2=8250, T2=8350,
        p1=0.35, p2=0.85  # 第一阶段难触发但需要高成功率补偿
    )
    if "error" in result3:
        print(f"错误: {result3['error']}")
    else:
        for k, v in result3.items():
            print(f"{k:>30}: {v}")