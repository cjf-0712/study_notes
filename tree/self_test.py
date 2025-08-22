# display_performance_evaluator.py

import numpy as np
from datetime import datetime


class DisplayPerformanceSystem:
    """
    显示器工况综合评估系统
    """

    def __init__(self, initial_luminance=100000):
        """
        初始化显示器评估系统
        initial_luminance: 初始亮度基准值 (cd/m²)
        """
        self.initial_luminance = initial_luminance
        self.performance_log = []
        self.evaluation_history = []

    def evaluate_display_performance(
            self,
            luminance_change,  # 亮度变动 (+增加/-减少)
            refresh_count,  # 刷新次数
            operating_temp,  # 工作温度 (°C)
            power_stability,  # 电源稳定性 (0-10)
            initial_state,  # 初始显示状态 (0-10)
            humidity_factor  # 环境湿度 (%RH)
    ) -> dict:
        """
        综合评价显示器在当前环境下的性能表现
        输出0-100分性能评分
        """
        # 记录评估时间
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 1. 显示效果分析 (权重40%)
        luminance_score = self._calculate_luminance_score(luminance_change)

        # 2. 刷新稳定性分析 (权重20%)
        refresh_score = self._calculate_refresh_score(refresh_count)

        # 3. 环境适应性分析 (权重40%)
        environment_score = self._calculate_environment_score(
            operating_temp,
            power_stability,
            initial_state,
            humidity_factor
        )

        # 综合性能评分
        performance_score = (
                luminance_score * 0.40 +
                refresh_score * 0.20 +
                environment_score * 0.40
        )

        # 性能评级
        performance_rating = self._determine_rating(performance_score)

        # 记录评估结果
        result = {
            "timestamp": timestamp,
            "performance_score": round(performance_score, 1),
            "performance_rating": performance_rating,
            "luminance_analysis": luminance_score,
            "refresh_analysis": refresh_score,
            "env_analysis": environment_score,
            "operating_temp": operating_temp,
            "power_stability": power_stability,
            "initial_state": initial_state,
            "humidity_factor": humidity_factor,
            "luminance_change": luminance_change,
            "refresh_count": refresh_count
        }

        self.performance_log.append(result)
        self.evaluation_history.append(performance_score)

        return result

    def _calculate_luminance_score(self, change):
        """亮度变化稳定性评估"""
        # 变化百分比
        change_percent = (change / self.initial_luminance) * 100

        if change_percent >= 5:  # 亮度增长>5%
            return min(95 + (change_percent - 5) * 1, 100)
        elif change_percent > -1:  # -1%至+5%轻微波动
            return 85 + change_percent * 2
        elif change_percent > -3:  # -3%至-1%
            return 70 + (change_percent + 3) * 5
        else:  # >-3%的衰减
            return max(50 + (change_percent + 3) * 5, 0)

    def _calculate_refresh_score(self, count):
        """刷新频率稳定性评估"""
        ideal_range = 3  # 理想刷新频率基准

        if count == 0:  # 无刷新
            return 50
        elif 1 <= count <= 5:  # 最佳刷新区间
            return 95 - abs(count - ideal_range) * 5
        elif 6 <= count <= 8:  # 略高但可接受
            return 85 - (count - 5) * 5
        elif 9 <= count <= 12:  # 过度刷新
            return 75 - (count - 8) * 8
        else:  # 严重过度刷新
            return max(45 - (count - 12) * 5, 0)

    def _calculate_environment_score(self, temp, power, init_state, humidity):
        """环境适应性综合评估"""
        # 温度稳定性因子 (30%)
        # 理想温度22-25°C, 超过范围扣分
        temp_dev = max(abs(temp - 23.5) - 1, 0)
        temp_score = max(90 - temp_dev * 10, 0)

        # 电源稳定性因子 (30%)
        power_score = (10 - power) * 8  # 0-80分

        # 初始状态因子 (20%)
        state_score = init_state * 9  # 0-90分

        # 湿度适应性因子 (20%)
        # 理想湿度40-60%RH
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

    def _determine_rating(self, score):
        """显示器性能评级"""
        if score >= 90:
            return "卓越性能A++"
        elif score >= 85:
            return "优秀性能A+"
        elif score >= 80:
            return "良好性能A"
        elif score >= 75:
            return "稳定性能B+"
        elif score >= 70:
            return "可用性能B"
        else:
            return "需调试优化"

    def generate_performance_report(self, latest_result):
        """生成显示器性能报告"""
        report = f"""
显示设备性能评估报告
=======================
评估时间: {latest_result['timestamp']}
综合性能指数: {latest_result['performance_score']}/100
性能评级: {latest_result['performance_rating']}

详细分析:
1. 亮度稳定性: {latest_result['luminance_analysis']}/100
   - 亮度变动量: {latest_result['luminance_change']} cd/m²

2. 刷新稳定性: {latest_result['refresh_analysis']}/100
   - 信号刷新次数: {latest_result['refresh_count']}

3. 环境适应性: {latest_result['env_analysis']}/100
   - 工作温度: {latest_result['operating_temp']}°C
   - 电源稳定度: {latest_result['power_stability']}/10
   - 初始状态值: {latest_result['initial_state']}/10
   - 环境湿度: {latest_result['humidity_factor']}%RH
"""
        return report

    def log_performance_trend(self, days=7):
        """记录性能趋势（实际使用时应连接数据库）"""
        print(f"记录最近{days}天的性能趋势...")
        print("（实际应用中应保存到数据库或文件）")
        return "性能日志更新成功"


# 示例使用
if __name__ == "__main__":
    # 创建显示器评估系统
    display_analyzer = DisplayPerformanceSystem(initial_luminance=100000)

    # 模拟输入参数
    display_params = {
        "luminance_change": -10000,  # 亮度变化
        "refresh_count": 8,  # 刷新次数
        "operating_temp": 50,  # 工作温度
        "power_stability": 3.0,  # 电源稳定性
        "initial_state": 7.5,  # 初始状态
        "humidity_factor": 45.2  # 环境湿度
    }

    # 进行显示器性能评估
    evaluation = display_analyzer.evaluate_display_performance(**display_params)

    # 生成性能报告
    print(display_analyzer.generate_performance_report(evaluation))

    # 记录性能趋势
    #display_analyzer.log_performance_trend()