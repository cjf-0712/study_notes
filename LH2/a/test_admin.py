import ctypes
# 检测是否为管理员权限（Windows 专用）
is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
print(f"是否以管理员权限运行：{is_admin}")
