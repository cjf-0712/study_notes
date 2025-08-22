# -*- coding: utf-8 -*-
"""
CTP 评测登录测试脚本（适配 vn.py 4.1.0 + vnpy_ctp 6.6.9.* + Python 3.10）

要点：
1) 新版导入：from vnpy_ctp import CtpGateway
2) 初始化必须传 gateway_name，例如 CtpGateway(event_engine, "CTP")
3) 网关需要中文键；脚本支持：
   - 方案A：英文键 -> 自动映射到中文键（默认启用，避免拼错）
   - 方案B：直接用中文键（更直观稳定，可一键切换）
"""

import time
import signal
import sys
from typing import Dict, List

from vnpy_ctp import CtpGateway
from vnpy.trader.object import LogData, AccountData
from vnpy.trader.event import EVENT_LOG, EVENT_ACCOUNT
from vnpy.event.engine import EventEngine

# ====== 根据需要切换 ======
USE_EN_TO_ZH_MAPPING = True  # True=用英文->中文自动映射；False=直接用中文键
WAIT_SECONDS = 20            # 等待登录结果的秒数
HOLD_MINUTES = 11             # 登录成功后保持运行的分钟数
# ========================


class CtpTestClient:
    def __init__(self):
        """初始化测试客户端（vn.py 4.x + Python 3.13）"""
        # 事件引擎
        self.event_engine = EventEngine()
        self.event_engine.start()

        # CTP 网关（新版必须带 gateway_name）
        self.ctp_gateway = CtpGateway(self.event_engine, "CTP")

        # 注册事件回调
        self.event_engine.register(EVENT_LOG, self.on_log)
        self.event_engine.register(EVENT_ACCOUNT, self.on_account)

        # 登录状态
        self.login_success = False

        # 你的英文配置（评测系统参数）
        self.test_config_en: Dict[str, str] = {
            "app_id": "Client_cjf_1.0.0",
            "auth_code": "M8DZS2SYRVC4AYNK",
            "broker_id": "6666",
            "td_address": "tcp://61.186.254.131:42205",
            "md_address": "tcp://61.186.254.131:42213",
            "user_id": "12345678",
            "password": "CS123456",
        }

        # 若要直接用中文键，把 USE_EN_TO_ZH_MAPPING 设为 False，并改这里：
        self.test_config_zh: Dict[str, str] = {
            "用户名": "12345678",
            "密码": "CS123456",
            "经纪商代码": "6666",
            "交易服务器": "tcp://61.186.254.131:42205",
            "行情服务器": "tcp://61.186.254.131:42213",
            "产品名称": "Client_cjf_1.0.0",
            "授权编码": "M8DZS2SYRVC4AYNK",
        }

    # -------------------------
    # 事件回调
    # -------------------------
    def on_log(self, event):
        """日志回调"""
        log: LogData = event.data
        t = log.time.strftime("%H:%M:%S") if getattr(log, "time", None) else time.strftime("%H:%M:%S")
        print(f"[{t}] {log.msg}")
        if "登录成功" in str(log.msg):
            self.login_success = True

    def on_account(self, event):
        """账户信息回调"""
        account: AccountData = event.data
        print(f"账户信息：{account.accountid}，可用资金：{account.available}")

    # -------------------------
    # 配置映射
    # -------------------------
    @staticmethod
    def _build_ctp_setting_from_en(en_cfg: Dict[str, str], default_setting_keys: List[str]) -> Dict[str, str]:
        """
        将英文键的配置映射到 CTP 网关需要的中文键（依据网关的 default_setting 动态生成）。
        """
        map_en2zh_candidates = {
            "user_id": ["用户名", "投资者帐号", "账户ID"],
            "password": ["密码"],
            "broker_id": ["经纪商代码", "BrokerID"],
            "td_address": ["交易服务器", "交易地址", "TradingAddress"],
            "md_address": ["行情服务器", "行情地址", "MarketAddress"],
            "app_id": ["产品名称", "AppID", "产品信息"],
            "auth_code": ["授权编码", "授权码", "AuthCode"],
        }

        setting: Dict[str, str] = {}
        for zh_key in default_setting_keys:
            filled = False
            for en_key, zh_list in map_en2zh_candidates.items():
                if zh_key in zh_list and en_key in en_cfg:
                    setting[zh_key] = en_cfg[en_key]
                    filled = True
                    break
            if not filled:
                setting[zh_key] = ""  # 占位，方便你在打印中看到缺哪些
        return setting

    # -------------------------
    # 主流程
    # -------------------------
    def start_test(self, wait_seconds: int = 15, hold_minutes: int = 11):
        """开始登录测试"""
        print("=" * 64)
        print("开始期货评测系统登录测试（vn.py 4.0.0 + vnpy_ctp 6.6.9.* + Py3.10.6）")
        print(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        # 打印版本信息（可选）
        try:
            import pkg_resources
            v_vnpy = pkg_resources.get_distribution("vnpy").version
            v_ctp = pkg_resources.get_distribution("vnpy_ctp").version
            print(f"版本：vnpy={v_vnpy}, vnpy_ctp={v_ctp}")
        except Exception:
            pass
        print("=" * 64)

        # 输出网关声明的默认配置键（中文）
        default_setting = getattr(self.ctp_gateway, "default_setting", {})
        default_keys = list(default_setting.keys()) if isinstance(default_setting, dict) else []
        print("网关需要的配置键：", "、".join(default_keys) if default_keys else "(未声明)")

        # 方案A：英文->中文映射（默认）
        if USE_EN_TO_ZH_MAPPING:
            setting = self._build_ctp_setting_from_en(self.test_config_en, default_keys)
        else:
            # 方案B：直接中文字典（键必须与 default_setting 完全同名）
            # 若键名不一致，请按打印出来的 default_keys 修改 self.test_config_zh 的键
            setting = {k: self.test_config_zh.get(k, "") for k in default_keys}

        print("\n将使用的连接配置：")
        for k, v in setting.items():
            print(f" - {k}: {v}")

        print("\n正在连接服务器...")
        self.ctp_gateway.connect(setting)

        # 等待登录结果
        for i in range(wait_seconds):
            if self.login_success:
                break
            print(f"等待登录响应... {i + 1}/{wait_seconds}")
            time.sleep(1)

        print("\n" + "=" * 64)
        if self.login_success:
            print("✅ 登录成功！请通知期货公司技术部进行后台验证")
            print(f"建议保持程序运行至少 {hold_minutes} 分钟，以便采集设备信息")
            print("=" * 64)
            try:
                for _ in range(hold_minutes):
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\n收到手动终止信号，准备退出...")
        else:
            print("❌ 登录失败，请检查：")
            print("1）是否在 17:00 之后（评测通常要求）")
            print("2）网络是否正常、未使用代理/VPN")
            print("3）参数是否准确（授权编码/产品名称/服务器地址等）")
            print("4）确保使用物理机（非虚拟机/云服务器）")
            print("=" * 64)

        self.cleanup()

    def cleanup(self):
        """停止事件引擎等清理操作"""
        try:
            self.event_engine.stop()
        except Exception:
            pass


def _install_signal_handlers(client: CtpTestClient):
    def _handler(signum, frame):
        print(f"\n收到信号 {signum}，准备清理并退出...")
        client.cleanup()
        sys.exit(0)

    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


if __name__ == "__main__":
    print("⚠️ 重要提示：请在物理机上运行，关闭所有虚拟机/云服务器环境")
    input("确认无误后按回车键继续...")

    client = CtpTestClient()
    _install_signal_handlers(client)
    client.start_test(wait_seconds=WAIT_SECONDS, hold_minutes=HOLD_MINUTES)
