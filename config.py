# -*- coding: utf-8 -*-
"""
全局配置：相机、颜色阈值、识别参数、JC24B 无线串口参数。

按 ENTERPRIZE RM2027 内部赛规则手册 2.2.5：
  - AR 升级区有 4 个待升级矿石位，按摄像头视角从左到右一一对应
  - 绿色塑料板 = 空位(EMPTY)
  - 红色/蓝色塑料板 = 有矿石(OCCUPIED)，红/蓝取决于队伍所在半场
  - 单局内颜色板不会被移除
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------- 槽位状态
EMPTY = 0      # 绿色 -> 空位
OCCUPIED = 1   # 红色/蓝色 -> 有矿石
UNKNOWN = 2    # 未识别到有效色板

STATE_NAME = {EMPTY: "EMPTY", OCCUPIED: "OCCUPIED", UNKNOWN: "UNKNOWN"}

# 绘制用 BGR 颜色
STATE_BGR = {
    EMPTY: (60, 200, 60),      # 绿
    OCCUPIED: (60, 60, 230),   # 红
    UNKNOWN: (110, 110, 110),  # 灰
}


# ---------------------------------------------------------------- HSV 颜色阈值
# 格式: { 颜色名: [(low_hsv, high_hsv), ...] }
# 红色在 HSV 中跨越 0 度，需要两段区间。
#
# 实测（现场塑料板在灯光下往往偏浅）：
#   绿色板 H=61 S=49 V=216   -> 浅绿，饱和度只有 50 左右，Smin 必须放低到 30 才抓得到
#   蓝色板 H=101 S=125 V=240 -> 饱和度高，Smin 70 安全
# 所以绿色单独用较低的 Smin；若把三者统一设成 70，浅绿会整块漏检。
# 现场灯光变化较大，务必用 `python main.py --tune` 重新标定。
HSV_RANGES = {
    "green": [
        ((40, 30, 45), (85, 255, 255)),
    ],
    "red": [
        ((0, 60, 50), (12, 255, 255)),
        ((168, 60, 50), (180, 255, 255)),
    ],
    "blue": [
        ((88, 65, 45), (132, 255, 255)),
    ],
}

# 判定为"空位"的颜色集合；其余(红/蓝)一律判为"有矿石"
EMPTY_COLORS = {"green"}
OCCUPIED_COLORS = {"red", "blue"}


# ---------------------------------------------------------------- 相机
@dataclass
class CameraConfig:
    index: int = 0                 # 摄像头序号；也可用视频文件路径做离线调试
    width: int = 1280
    height: int = 720
    fps: int = 60
    # 检测在缩略图上进行以保证 >=30fps，绘制仍在原分辨率
    detect_width: int = 640
    fourcc: Optional[str] = None   # 例如 "MJPG"，部分摄像头用 MJPG 才能跑满帧率
    exposure: Optional[float] = None   # 手动曝光，如 -6；None 表示自动
    autofocus: bool = False


# ---------------------------------------------------------------- 识别参数
@dataclass
class DetectConfig:
    slot_count: int = 4

    # ---- 色板候选轮廓筛选 ----
    # 色板实际尺寸 160mm x 160mm，按现场距离设置一个大致的面积范围（像素^2，基于 detect 分辨率）
    min_area: int = 300
    max_area: int = 120000
    min_solidity: float = 0.55     # 轮廓面积 / 凸包面积，滤掉细长噪点
    aspect_range: Tuple[float, float] = (0.35, 2.8)

    # ---- 槽位锁定 ----
    # SEARCHING -> LOCKED：连续 lock_frames 帧稳定找到 4 个候选才锁定，避免误锁
    lock_frames: int = 8
    # 锁定后允许的最大中心漂移（检测分辨率下的像素），超过则视为跟踪丢失
    max_drift: int = 90
    # 连续 miss_frames 帧找不到 4 个候选则释放锁定重新全局搜索
    miss_frames: int = 30
    # 锁定框相对检测框的扩张系数
    roi_scale: float = 1.35

    # ---- 单帧颜色判定 ----
    min_coverage: float = 0.12     # ROI 内该颜色像素占比低于此值视为无有效色板
    morph_kernel: int = 5          # 形态学核大小(奇数)
    blur_kernel: int = 5           # 高斯模糊核大小(奇数)

    # ---- 时间域滤波 ----
    vote_window: int = 9           # 众数投票窗口帧数
    # ---- 规则 3.4.2：维持 5 秒无目标丢失/误识别才算识别成功 ----
    stable_seconds: float = 5.0

    # ---- 手动 ROI 模式 ----
    roi_file: Optional[str] = None  # json: {"slots": [[x,y,w,h], ...]}  基于检测分辨率
    manual_mode: bool = False


# ---------------------------------------------------------------- JC24B 无线串口
@dataclass
class RadioConfig:
    enabled: bool = False      # True 才真正打开串口；False 为 dry-run（只打印，不占串口）
    port: str = "COM3"         # Windows: COM3 / Linux: /dev/ttyUSB0
    baudrate: int = 115200     # JC24B 常用波特率，需与模块 AT 配置一致
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 0.2
    # 状态变化立即发送；无变化时按 heartbeat_ms 周期重发（对端可做掉线检测）
    heartbeat_ms: int = 500
    # 发送失败连续失败次数达到该值则尝试重连
    reconnect_after_fail: int = 5
    # 帧尾是否附加 CRLF（部分透传模块/上位机按行解析）
    append_crlf: bool = False


# ---------------------------------------------------------------- 协议常量
# JC24B 为透明传输的无线串口模块，本端只负责组帧，对端(AR 主控)按同一协议解析。
class Protocol:
    HEAD = 0xAA55
    HEAD_BYTES = b"\xAA\x55"
    CMD_SLOT_STATE = 0x10   # 4 个槽位的颜色识别结果
    RESERVED = 0x00         # 预留：后续接入角度识别(+/-30°)结果
    FRAME_LEN = 9           # AA 55 CMD S1 S2 S3 S4 SEQ CRC

    @staticmethod
    def crc8(data: bytes) -> int:
        """CRC-8/ATM (poly=0x07)，覆盖 CMD 到 SEQ。"""
        crc = 0x00
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        return crc
