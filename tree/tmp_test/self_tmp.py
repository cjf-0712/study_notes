# display_performance_evaluator.py

import os
import csv
from datetime import datetime, timedelta


class DisplayPerformanceSystem:
    """
    显示器工况综合评估系统（带数据存储功能）
    """

    def __init__(self, initial_luminance=100000, storage_path=None):
        """
        初始化显示器评估系统
        initial_luminance: 初始亮度基准值 (cd/m²)
        storage_path: 数据存储路径 (默认: 当前目录下的performance_data文件夹)
        """
        self.initial_luminance = initial_luminance
        self.performance_log = []
        self.evaluation_history = []

        # 设置存储路径
        if storage_path is None:
            # 默认存储到当前目录下的performance_data文件夹
            self.storage_path = os.path.join(os.getcwd(), "performance_data")
        else:
            self.storage_path = storage_path

        # 创建存储目录（如果不存在）
        os.makedirs(self.storage_path, exist_ok=True)

    def evaluate_display_performance(
            self,
            luminance_change,  # 亮度变动 (+增加/-减少)
            refresh_count,  # 刷新次数
            operating_temp,  # 工作温度 (°C)
            power_stability,  # 电源稳定性 (0-10)
            initial_state,  # 初始显示状态 (0-10)
            humidity_factor,  # 环境湿度 (%RH)
            save_to_file=False  # 是否立即保存到文件
    ) -> dict:
        # ... [前面的代码不变，同上] ...

        # 保存到日志
        self.performance_log.append(result)
        self.evaluation_history.append(performance_score)

        # 如果需要，立即保存到文件
        if save_to_file:
            self.save_performance_log()

        return result

    # ... [其他方法不变，同上] ...

    def save_performance_log(self, filename=None):
        """将性能日志保存到CSV文件"""
        if not self.performance_log:
            return {"status": "warning", "message": "无性能数据可保存"}

        # 确定文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"display_performance_{timestamp}.csv"

        file_path = os.path.join(self.storage_path, filename)

        # CSV文件头
        fieldnames = [
            "timestamp", "performance_score", "performance_rating",
            "luminance_analysis", "refresh_analysis", "env_analysis",
            "operating_temp", "power_stability", "initial_state",
            "humidity_factor", "luminance_change", "refresh_count"
        ]

        # 写入CSV文件
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.performance_log)

            return {"status": "success", "message": f"数据已保存至: {file_path}", "record_count": len(self.performance_log)}
        except Exception as e:
            return {"status": "error", "message": f"保存失败: {str(e)}"}

    def load_performance_log(self, filename):
        """从CSV文件加载性能日志"""
        file_path = os.path.join(self.storage_path, filename)

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"文件不存在: {file_path}"}

        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                self.performance_log = []
                for row in reader:
                    # 转换数据类型
                    converted_row = {}
                    for key, value in row.items():
                        # 特殊处理时间戳
                        if key == "timestamp":
                            converted_row[key] = value
                        else:
                            try:
                                converted_row[key] = float(value) if '.' in value else int(value)
                            except:
                                converted_row[key] = value

                    self.performance_log.append(converted_row)

            # 重建评估历史
            self.evaluation_history = [log["performance_score"] for log in self.performance_log]

            return {"status": "success", "message": f"已加载 {len(self.performance_log)} 条记录", "file_path": file_path}
        except Exception as e:
            return {"status": "error", "message": f"加载失败: {str(e)}"}

    def get_recent_performance_trend(self, days=7):
        """获取最近N天的性能趋势数据"""
        if not self.performance_log:
            return {"status": "warning", "message": "无性能数据可用"}

        # 计算日期范围
        cutoff_date = datetime.now() - timedelta(days=days)

        # 筛选符合条件的记录
        recent_data = []
        for record in self.performance_log:
            record_date = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M")
            if record_date > cutoff_date:
                recent_data.append(record)

        # 计算趋势统计
        if not recent_data:
            return {"status": "warning", "message": f"无最近{days}天的数据"}

        scores = [data["performance_score"] for data in recent_data]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        # 生成趋势分析报告
        trend_report = f"""
显示器性能趋势分析报告 ({days}天)
=================================
分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}
记录数量: {len(recent_data)}
性能分数范围: {min_score:.1f} - {max_score:.1f} (平均: {avg_score:.1f})

主要指标趋势:
- 亮度稳定性: {min([r['luminance_analysis'] for r in recent_data]):.1f} 至 {max([r['luminance_analysis'] for r in recent_data]):.1f}
- 刷新稳定性: {min([r['refresh_analysis'] for r in recent_data]):.1f} 至 {max([r['refresh_analysis'] for r in recent_data]):.1f}
- 环境适应性: {min([r['env_analysis'] for r in recent_data]):.1f} 至 {max([r['env_analysis'] for r in recent_data]):.1f}

建议优化方向:
"""
        # 添加基于分析的优化建议
        if avg_score < 75:
            trend_report += "> 设备性能不稳定，建议进行环境优化和设备调试\n"
        elif min_score < 65:
            trend_report += "> 存在性能低谷期，建议分析特定条件进行优化\n"
        else:
            trend_report += "> 性能表现稳定，保持当前环境设置即可\n"

        return {
            "status": "success",
            "report": trend_report,
            "record_count": len(recent_data),
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score
        }

    def list_stored_files(self):
        """列出存储目录中的所有CSV文件"""
        try:
            all_files = os.listdir(self.storage_path)
            csv_files = [f for f in all_files if f.lower().endswith('.csv')]

            # 按修改时间排序
            csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.storage_path, f)), reverse=True)

            return {"status": "success", "files": csv_files}
        except Exception as e:
            return {"status": "error", "message": f"读取目录失败: {str(e)}"}

    def clear_performance_log(self):
        """清除内存中的性能日志"""
        self.performance_log = []
        self.evaluation_history = []
        return {"status": "success", "message": "内存日志已清除"}


# 示例使用
if __name__ == "__main__":
    # 创建显示器评估系统并指定存储路径
    custom_path = "F:/FPGA/data"  # 可以修改为您需要的任何路径
    display_analyzer = DisplayPerformanceSystem(
        initial_luminance=100000,
        storage_path=custom_path
    )

    # 添加多条数据记录（模拟7天数据）
    for day in range(1, 8):
        # 生成参数（模拟数据）
        display_params = {
            "luminance_change": 3000 + day * 100,
            "refresh_count": 8 if day % 2 == 0 else 6,
            "operating_temp": 22.0 + day * 0.5,
            "power_stability": 3.5 if day < 4 else 2.8,
            "initial_state": 7.5 - (day % 3) * 0.5,
            "humidity_factor": 45.0 + day * 1.5,
            "save_to_file": False  # 最后统一保存
        }

        # 调整时间戳（模拟过去日期）
        result = display_analyzer.evaluate_display_performance(**display_params)
        # 修改时间戳为模拟日期
        past_date = (datetime.now() - timedelta(days=7 - day)).strftime("%Y-%m-%d %H:%M")
        display_analyzer.performance_log[-1]["timestamp"] = past_date

    # 保存模拟数据
    save_result = display_analyzer.save_performance_log("simulated_performance_data.csv")
    print(save_result["message"])

    # 查询存储目录中的文件
    file_list = display_analyzer.list_stored_files()
    print("\n存储目录中的性能文件:")
    for i, file in enumerate(file_list["files"], 1):
        print(f"{i}. {file}")

    # 清空内存日志
    display_analyzer.clear_performance_log()

    # 从文件加载数据
    if file_list["files"]:
        load_file = file_list["files"][0]  # 加载最新文件
        load_result = display_analyzer.load_performance_log(load_file)
        print(f"\n{load_result['message']}")

        # 分析最近7天的性能趋势
        trend_report = display_analyzer.get_recent_performance_trend(days=7)
        print("\n性能趋势分析报告:")
        print(trend_report["report"])

    # 单独评估当前性能
    current_params = {
        "luminance_change": 3200,
        "refresh_count": 7,
        "operating_temp": 23.5,
        "power_stability": 3.2,
        "initial_state": 7.8,
        "humidity_factor": 48.0
    }
    current_evaluation = display_analyzer.evaluate_display_performance(**current_params)
    print("\n当前性能评估报告:")
    print(display_analyzer.generate_performance_report(current_evaluation))