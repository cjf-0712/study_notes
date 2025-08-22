#ps
"""
# CPU ID
Get-CimInstance Win32_Processor | Select-Object Name, ProcessorId

# BIOS 序列号
Get-CimInstance Win32_BIOS | Select-Object SerialNumber, SMBIOSBIOSVersion

# 主板序列号
Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer, Product, SerialNumber

# 磁盘序列号（旧接口）
Get-CimInstance Win32_DiskDrive | Select-Object Model, SerialNumber
pip install "vnpy_ctp==6.6.9.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
# 磁盘序列号（新接口，更可靠）
Get-PhysicalDisk | Select-Object FriendlyName, SerialNumber, BusType, Size


"""