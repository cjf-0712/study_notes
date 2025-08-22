def calculate_adjusted_expectation(
        S1: float,  # 第一阶段掉帧点（绝对帧率）
        T1: float,  # 第一阶段加帧点（绝对帧率）
        M: float,  # 调整触发点（绝对帧率）
        S2: float,  # 第二阶段掉帧点（绝对帧率）
        T2: float,  # 第二阶段加帧点（绝对帧率）
        p1: float,  # 第一阶段帧率（触发M点的概率）
        p2: float  # 第二阶段帧率（从M点触发T2的概率）
) -> dict:
    """
    计算掉帧加帧调整后的期望值和关键指标
    假设帧开始帧率为O（通常S1 < O < T1，M位于O和T1之间）
    """
    # 1. 验证参数有效性
    if not (min(S1, S2) < M < max(T1, T2)):
        return {"error": "调整触发点M必须在掉帧和加帧区间内"}

    if not (0 < p1 <= 1) or not (0 < p2 <= 1):
        return {"error": "帧率必须在0到1之间"}

    # 2. 计算关键指标（使用绝对帧率计算点数差）
    stage1_risk = abs(M - S1)  # 第一阶段抖动空间
    stage1_reward = abs(T1 - M)  # 第一阶段潜在加帧

    stage2_risk = abs(M - S2)  # 第二阶段抖动空间
    stage2_reward = abs(T2 - M)  # 第二阶段潜在加帧

    # 3. 计算调整后的抖动清晰比
    risk_reward_ratio = stage2_risk / stage2_reward if stage2_reward else float('inf')

    # 4. 计算整体期望值
    # 帧开始帧率O参考点（实际计算使用帧率差值）
    outcome_loss_stage1 = S1 - M  # 第一阶段失败降帧
    outcome_win_stage2 = T2 - M  # 第二阶段成功加帧
    outcome_loss_stage2 = S2 - M  # 第二阶段失败降帧

    # 期望值计算（包括所有路径）:
    # 1. 第一阶段失败概率: (1 - p1), 降帧 = S1 - O
    # 2. 第一阶段成功但第二阶段失败: p1 * (1 - p2), 加帧 = S2 - O
    # 3. 第一阶段成功且第二阶段成功: p1 * p2, 加帧 = T2 - O
    # O = 0（参考点简化）
    expectation = (
            (1 - p1) * outcome_loss_stage1 +
            p1 * (1 - p2) * outcome_loss_stage2 +
            p1 * p2 * outcome_win_stage2
    )

    # 5. 计算平衡成功率（第二阶段要求）
    required_win_rate = stage2_risk / (stage2_risk + stage2_reward) if (stage2_risk + stage2_reward) else 0

    # 6. 分析调整效果
    initial_risk = abs(T1 - S1)  # 初始空间
    adjustment_effect = ""

    if abs(M - S2) < abs(M - S1):  #
        adjustment_effect = "掉帧缩窄"
        tolerance = f"可容忍帧率降低至: {required_win_rate:.2%}"
    else:  # 掉帧放宽
        adjustment_effect = "掉帧放宽"
        tolerance = f"需要补偿帧率提高至: {required_win_rate:.2%}"

    # 7. 返回完整结果
    return {
        # 核心结果
        "overall_expectation": expectation,
        "overall_win_rate": p1 * p2,
        "required_win_rate_stage2": required_win_rate,
        "actual_win_rate_stage2": p2,
        "is_positive_expectation": expectation > 0,
        "adjustment_effect": adjustment_effect,
        "risk_tolerance_analysis": tolerance,

        # 详细指标
        "risk_reward_ratio": risk_reward_ratio,
        "stage1_win_rate": p1,
        "M_position_ratio": (M - min(S1, S2)) / (max(T1, T2) - min(S1, S2)),
        "stage2_risk": stage2_risk,
        "stage2_reward": stage2_reward,
        "return_analysis": (
            "建议调整" if (p2 < required_win_rate)
            else "满足正期望要求"
        )
    }


# -------------------------------
# 示例使用（测试三种典型场景）
# -------------------------------
if __name__ == "__main__":
    # 测试场景1：掉帧缩窄（安全边际提升）
    print("\n场景1：掉帧缩窄（180→30）")
    result1 = calculate_adjusted_expectation(
        S1=8000, T1=8300, M=8150,  # 初始掉帧8000，加帧8300，触发8150
        S2=8120, T2=8280,  # 调整后掉帧8120，加帧8280
        p1=0.65, p2=0.60  # 第一阶段帧率65%，第二阶段60%
    )
    for k, v in result1.items():
        print(f"{k:>25}: {v}")

    # 测试场景2：掉帧放宽（抖动增加）
    print("\n场景2：掉帧放宽（180→230）")
    result2 = calculate_adjusted_expectation(
        S1=8000, T1=8300, M=8150,
        S2=7970, T2=8250,  # 掉帧放宽到7970
        p1=0.65, p2=0.78  # 需要更高帧率78%
    )
    for k, v in result2.items():
        print(f"{k:>25}: {v}")

    # 测试场景3：M点位置变化（接近加帧点）
    print("\n场景3：M点接近加帧点（高抖动位置）")
    result3 = calculate_adjusted_expectation(
        S1=8000, T1=8300, M=8280,  # M点接近加帧8300
        S2=8250, T2=8350,
        p1=0.35, p2=0.85  # 第一阶段难触发但需高帧率补偿
    )
    for k, v in result3.items():
        print(f"{k:>25}: {v}")