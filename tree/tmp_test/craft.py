# display_performance_evaluator.py
import os
import csv
from datetime import datetime
import numpy as np


class DisplayPerformanceSystem:
    """
    显示器工况评估系统（支持三天亮度变化输入）
    """

    def __init__(self, storage_path="F:\\FPGA\\data", initial_luminance=100000):
        """
        初始化显示器评估系统
        storage_path: 数据存储路径 - 设置为您指定的路径
        initial_luminance: 初始亮度基准值
        """
        self.initial_luminance = initial_luminance
        self.storage_path = storage_path

        # 创建存储目录（如果不存在）
        os.makedirs(storage_path, exist_ok=True)

    def evaluate_display_performance(
            self,
            luminance_change_today,  # 今天的亮度变化
            luminance_change_yesterday,  # 昨天的亮度变化
            luminance_change_day_before,  # 前天的亮度变化
            refresh_count,  # 刷新次数
            operating_temp,  # 工作温度 (°C)
            power_stability,  # 电源稳定性 (0-10)
            initial_state,  # 初始显示状态 (0-10)
            humidity_factor,  # 环境湿度 (%RH)
            save_to_file=True,  # 是否保存结果
            filename=None  # 自定义文件名
    ) -> dict:
        """
        综合评价显示器性能并生成报告，使用三天亮度变化值
        """
        # 1. 计算亮度变化分数（使用三天数据）
        luminance_score = self._calculate_luminance_score(
            luminance_change_today,
            luminance_change_yesterday,
            luminance_change_day_before
        )

        # 2. 计算其他分数
        refresh_score = self._calculate_refresh_score(refresh_count)
        env_score = self._calculate_environment_score(
            operating_temp, power_stability, initial_state, humidity_factor
        )

        # 3. 综合性能评分
        performance_score = (
                luminance_score * 0.40 +
                refresh_score * 0.20 +
                env_score * 0.40
        )

        # 4. 性能评级
        performance_rating = self._get_performance_rating(performance_score)

        # 5. 分析亮度趋势
        trend_description = self._get_luminance_trend_description(
            luminance_change_today,
            luminance_change_yesterday,
            luminance_change_day_before
        )

        # 6. 创建结果字典
        result_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "performance_score": round(performance_score, 1),
            "performance_rating": performance_rating,
            "luminance_analysis": luminance_score,
            "refresh_analysis": refresh_score,
            "env_analysis": env_score,
            "operating_temp": operating_temp,
            "power_stability": power_stability,
            "initial_state": initial_state,
            "humidity_factor": humidity_factor,
            "luminance_today": luminance_change_today,
            "luminance_yesterday": luminance_change_yesterday,
            "luminance_day_before": luminance_change_day_before,
            "refresh_count": refresh_count,
            "luminance_trend": trend_description
        }

        # 7. 保存结果
        if save_to_file:
            self._save_performance(result_data, filename)

        return result_data

    def _calculate_luminance_score(self, today_change, yesterday_change, day_before_change):
        """计算亮度变化稳定性评分，使用三天数据"""
        # 基本亮度变化分数（今天）
        base_score = self._get_base_luminance_score(today_change)

        # 趋势分析惩罚因子
        trend_penalty = self._calculate_trend_penalty(
            today_change, yesterday_change, day_before_change
        )

        # 应用惩罚（如果适用）
        final_score = max(base_score * (1 - trend_penalty), 0)

        return min(round(final_score, 2), 100)

    def _get_base_luminance_score(self, change):
        """获取基本亮度分数（不考虑趋势）"""
        change_percent = (change / self.initial_luminance) * 100

        if change_percent >= 5:
            return min(95 + (change_percent - 5) * 1, 100)
        elif change_percent > -1:
            return 85 + change_percent * 2
        elif change_percent > -3:
            return 70 + (change_percent + 3) * 5
        else:
            return max(50 + (change_percent + 3) * 5, 0)

    def _calculate_trend_penalty(self, today_change, yesterday_change, day_before_change):
        """计算连续单方向变化的惩罚因子"""
        # 将三天变化值转换为列表
        changes = [day_before_change, yesterday_change, today_change]

        # 检查是否都是同向变化（正或负）
        all_positive = all(c > 0 for c in changes)
        all_negative = all(c < 0 for c in changes)

        # 计算连续同方向的变化次数
        consecutive_count = 0
        last_sign = 0  # 0表示没有方向，1表示正，-1表示负

        for c in changes:
            if c > 0:
                current_sign = 1
            elif c < 0:
                current_sign = -1
            else:  # 变化为0
                current_sign = 0

            if current_sign == last_sign or last_sign == 0:
                if current_sign != 0:  # 忽略0变化的情况
                    consecutive_count += 1
            else:
                consecutive_count = 1  # 方向变化，重新开始计数

            last_sign = current_sign

        # 如果少于2天同向变化，不施加惩罚
        if consecutive_count < 2 or (not all_positive and not all_negative):
            return 0

        # 计算惩罚因子
        penalty_factor = min(0.1 * (consecutive_count - 1), 0.3)  # 最大惩罚30%

        # 额外惩罚：如果连续变化在加速
        if consecutive_count >= 2:
            changes_abs = [abs(c) for c in changes]  # 绝对值变化

            # 检查是否有加速趋势 (今天 > 昨天 > 前天)
            is_accelerating = (
                    changes_abs[2] > changes_abs[1] and
                    changes_abs[1] > changes_abs[0]
            )

            if is_accelerating:
                penalty_factor = min(penalty_factor + 0.15, 0.45)  # 加速趋势额外惩罚

        return penalty_factor

    def _get_luminance_trend_description(self, today_change, yesterday_change, day_before_change):
        """获取亮度变化趋势描述（使用三天数据）"""
        changes = [day_before_change, yesterday_change, today_change]

        # 描述三天的变化方向
        day_names = ["前天", "昨天", "今天"]
        descriptions = []

        for i, change in enumerate(changes):
            if change > 0:
                descriptions.append(f"{day_names[i]}亮度增加了{abs(change)}")
            elif change < 0:
                descriptions.append(f"{day_names[i]}亮度减少了{abs(change)}")
            else:
                descriptions.append(f"{day_names[i]}亮度无变化")

        # 整体趋势判断
        signs = [1 if c > 0 else -1 if c < 0 else 0 for c in changes]

        if signs == [1, 1, 1] or signs == [-1, -1, -1]:
            trend_intensity = "强烈"
        elif signs.count(signs[0]) >= 2:
            trend_intensity = "中等"
        else:
            trend_intensity = "波动"

        direction = ""
        if all(sign > 0 for sign in signs if sign != 0):
            direction = "连续增强"
        elif all(sign < 0 for sign in signs if sign != 0):
            direction = "连续减弱"

        # 组合描述
        trend_desc = "; ".join(descriptions)
        if direction:
            trend_desc += f" → 整体呈现{trend_intensity}的{direction}趋势"
        else:
            trend_desc += " → 无一致趋势"

        return trend_desc

    # 刷新稳定性和环境适应性的计算方法保持不变
    def _calculate_refresh_score(self, count):
        """计算刷新频率稳定性评分"""
        if count == 0:
            return 50
        elif 1 <= count <= 5:
            return 95 - abs(count - 3) * 5
        elif 6 <= count <= 8:
            return 85 - (count - 5) * 5
        elif 9 <= count <= 12:
            return 75 - (count - 8) * 8
        else:
            return max(45 - (count - 12) * 5, 0)

    def _calculate_environment_score(self, temp, power, init_state, humidity):
        """计算环境适应性评分"""
        # 温度稳定性
        temp_dev = max(abs(temp - 23.5) - 1, 0)
        temp_score = max(90 - temp_dev * 10, 0)

        # 电源稳定性
        power_score = (10 - power) * 8  # 0-80分

        # 初始状态
        state_score = init_state * 9  # 0-90分

        # 湿度适应性
        humidity_dev = min(
            abs(humidity - 40),
            abs(humidity - 60)
        ) if humidity < 40 or humidity > 60 else 0
        humidity_score = max(85 - humidity_dev * 5, 0)

        # 综合环境分
        env_score = (
                temp_score * 0.3 +
                power_score * 0.3 +
                state_score * 0.2 +
                humidity_score * 0.2
        )

        return min(env_score, 100)

    def _get_performance_rating(self, score):
        """根据分数获取性能评级"""
        if score >= 90: return "卓越性能A++"
        if score >= 85: return "优秀性能A+"
        if score >= 80: return "良好性能A"
        if score >= 75: return "稳定性能B+"
        if score >= 70: return "可用性能B"
        return "需调试优化"

    def _save_performance(self, result_data, filename=None):
        """保存性能数据到CSV文件"""
        if filename is None:
            filename = f"display_performance_{datetime.now().strftime('%Y%m%d')}.csv"

        file_path = os.path.join(self.storage_path, filename)

        # 检查文件是否存在，决定是否添加标题
        write_header = not os.path.exists(file_path)

        try:
            # CSV字段
            fieldnames = [
                "timestamp", "performance_score", "performance_rating",
                "luminance_analysis", "refresh_analysis", "env_analysis",
                "operating_temp", "power_stability", "initial_state",
                "humidity_factor", "luminance_today", "luminance_yesterday",
                "luminance_day_before", "refresh_count", "luminance_trend"
            ]

            # 写入或追加到文件
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if write_header:
                    writer.writeheader()

                writer.writerow(result_data)

            return {"status": "success", "file_path": file_path}
        except Exception as e:
            return {"status": "error", "message": f"保存失败: {str(e)}"}

    def get_performance_report(self, result_data):
        """生成性能报告文本"""
        return f"""
显示设备性能评估报告
=======================
评估时间: {result_data['timestamp']}
综合性能指数: {result_data['performance_score']}/100
性能评级: {result_data['performance_rating']}

亮度分析:
- 前天变化: {result_data['luminance_day_before']} cd/m²
- 昨天变化: {result_data['luminance_yesterday']} cd/m²
- 今天变化: {result_data['luminance_today']} cd/m²
- 亮度稳定性分数: {result_data['luminance_analysis']}/100
- 趋势分析: {result_data['luminance_trend']}

刷新分析:
- 刷新稳定性分数: {result_data['refresh_analysis']}/100
- 信号刷新次数: {result_data['refresh_count']}

环境适应性:
- 综合分数: {result_data['env_analysis']}/100
- 工作温度: {result_data['operating_temp']}°C
- 电源稳定度: {result_data['power_stability']}/10
- 初始状态值: {result_data['initial_state']}/10
- 环境湿度: {result_data['humidity_factor']}%RH
"""


# 示例使用
if __name__ == "__main__":
    print("显示器性能评估系统启动...")

    # 创建评估系统，使用您指定的存储路径
    analyzer = DisplayPerformanceSystem(storage_path="F:\\FPGA\\data")

    # 模拟连续三天的亮度变化
    day_before_change = 200  # 前天变化
    yesterday_change = 500  # 昨天变化
    today_change = -690  # 今天变化

    # 其他参数保持不变
    evaluation_data = {
        "luminance_change_today": today_change,
        "luminance_change_yesterday": yesterday_change,
        "luminance_change_day_before": day_before_change,
        "refresh_count": 10,
        "operating_temp": 23.5,
        "power_stability": 4.2,
        "initial_state": 8.0,
        "humidity_factor": 60.0,
        "save_to_file": True
    }

    # 评估性能
    result = analyzer.evaluate_display_performance(**evaluation_data)

    # 生成报告
    report = analyzer.get_performance_report(result)

    # 输出结果
    print("\n性能评估报告:")
    print(report)

    # 显示存储位置
    print(f"数据已保存至: F:\\FPGA\\data")