# ioctldisk_sn.py  管理员运行
import ctypes, struct
import win32file

# 常量
IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
StorageDeviceProperty = 0        # 要查询“设备”属性
PropertyStandardQuery   = 0

class STORAGE_PROPERTY_QUERY(ctypes.Structure):
    _fields_ = [
        ("PropertyId", ctypes.c_int),
        ("QueryType", ctypes.c_int),
        ("AdditionalParameters", ctypes.c_byte * 1),
    ]

def _read_c_string(buf: bytes, offset: int) -> str | None:
    if offset in (0, 0xFFFFFFFF) or offset >= len(buf):
        return None
    # 取以 \0 结尾的 ANSI 字符串
    end = buf.find(b"\x00", offset)
    if end == -1:
        end = len(buf)
    try:
        return buf[offset:end].decode("ansi", errors="ignore").strip()
    except Exception:
        return None

def get_serial_via_ioctl(physical_index: int = 0) -> dict:
    path = fr"\\.\PhysicalDrive{physical_index}"
    h = win32file.CreateFile(
        path,
        win32file.GENERIC_READ,
        win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
        None, win32file.OPEN_EXISTING, 0, None
    )
    try:
        q = STORAGE_PROPERTY_QUERY()
        q.PropertyId = StorageDeviceProperty
        q.QueryType  = PropertyStandardQuery

        # 读 4KB 足够覆盖描述符与附加属性
        out = win32file.DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, q, 4096)

        # 解析 STORAGE_DEVICE_DESCRIPTOR 头部
        # DWORD Version, DWORD Size, BYTE DevType, BYTE DevTypeMod, BOOL Removable, BOOL Queueing,
        # DWORD VendorIdOffset, DWORD ProductIdOffset, DWORD ProductRevOffset, DWORD SerialNumberOffset,
        # DWORD BusType, DWORD RawLen
        hdr = struct.unpack_from("<IIBB??IIIIII", out, 0)
        serial_off = hdr[9]

        serial = _read_c_string(out, serial_off)
        vendor = _read_c_string(out, hdr[6])
        product = _read_c_string(out, hdr[7])
        rev = _read_c_string(out, hdr[8])

        return {
            "path": path,
            "serial": serial,
            "vendor": vendor,
            "product": product,
            "revision": rev,
            "buf_len": len(out)
        }
    finally:
        h.Close()

if __name__ == "__main__":
    for i in range(0, 4):  # 依次试 0..3 号物理盘
        try:
            info = get_serial_via_ioctl(i)
            print(f"[{info['path']}] SN={info['serial']}  Vendor={info['vendor']}  "
                  f"Product={info['product']}  Rev={info['revision']}  (buf={info['buf_len']})")
        except Exception as e:
            # 没有该物理盘或权限问题会在这里抛
            print(f"[\\\\.\\PhysicalDrive{i}] 失败：{e}")
