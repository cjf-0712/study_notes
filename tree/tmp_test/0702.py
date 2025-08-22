# display_performance_evaluator.py
import os
import csv
from datetime import datetime


class DisplayPerformanceSystem:
    """
    显示器工况评估系统（简化版）
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
            luminance_change,  # 亮度变动 (+增加/-减少)
            refresh_count,  # 刷新次数
            operating_temp,  # 工作温度 (°C)
            power_stability,  # 电源稳定性 (0-10)
            initial_state,  # 初始显示状态 (0-10)
            humidity_factor,  # 环境湿度 (%RH)
            save_to_file=True,  # 是否保存结果
            filename=None  # 自定义文件名
    ) -> dict:
        """
        综合评价显示器性能并生成报告
        """
        # 1. 计算性能分数
        luminance_score = self._calculate_luminance_score(luminance_change)
        refresh_score = self._calculate_refresh_score(refresh_count)
        env_score = self._calculate_environment_score(
            operating_temp, power_stability, initial_state, humidity_factor
        )

        # 综合性能评分
        performance_score = (
                luminance_score * 0.40 +
                refresh_score * 0.20 +
                env_score * 0.40
        )

        # 性能评级
        performance_rating = self._get_performance_rating(performance_score)

        # 创建结果字典
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
            "luminance_change": luminance_change,
            "refresh_count": refresh_count
        }

        # 保存结果
        if save_to_file:
            self._save_performance(result_data, filename)

        return result_data

    def _calculate_luminance_score(self, change):
        """计算亮度变化稳定性评分"""
        change_percent = (change / self.initial_luminance) * 100

        if change_percent >= 5:
            return min(95 + (change_percent - 5) * 1, 100)
        elif change_percent > -1:
            return 85 + change_percent * 2
        elif change_percent > -3:
            return 70 + (change_percent + 3) * 5
        else:
            return max(50 + (change_percent + 3) * 5, 0)

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
                "humidity_factor", "luminance_change", "refresh_count"
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

详细分析:
1. 亮度稳定性: {result_data['luminance_analysis']}/100
   - 亮度变动量: {result_data['luminance_change']} cd/m²

2. 刷新稳定性: {result_data['refresh_analysis']}/100
   - 信号刷新次数: {result_data['refresh_count']}

3. 环境适应性: {result_data['env_analysis']}/100
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

    # 示例评估数据
    evaluation_data = {
        "luminance_change": 3500,  # 亮度变化
        "refresh_count": 5,  # 刷新次数
        "operating_temp": 23.5,  # 工作温度
        "power_stability": 4.2,  # 电源稳定性
        "initial_state": 8.0,  # 初始状态
        "humidity_factor": 45.0  # 环境湿度
    }

    # 评估性能
    result = analyzer.evaluate_display_performance(**evaluation_data)

    # 生成报告
    report = analyzer.get_performance_report(result)

    # 输出结果
    print("\n性能评估完成:")
    print(f"- 综合评分: {result['performance_score']}/100")
    print(f"- 性能评级: {result['performance_rating']}")

    print("\n详细报告:")
    print(report)

    # 显示存储位置
    print(f"数据已保存至: F:\\FPGA\\data")