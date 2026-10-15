# -*- coding: utf-8 -*-
"""
JC24B 无线串口模块发送接口。

JC24B 是 2.4GHz 透明传输无线串口模块，串口侧就是普通 UART：
    CV 端(PC) --USB转串口--> JC24B ))) 无线 ((( JC24B --UART--> AR 主控(STM32 等)

使用方式：
    link = JC24BLink(RadioConfig(port="COM3", baudrate=115200, enabled=True))
    link.open()
    link.send_slots([EMPTY, OCCUPIED, EMPTY, OCCUPIED])   # 从左到右 4 个槽位
    link.close()

要点：
  1. enabled=False 时为 dry-run，只打印不占串口，方便没有模块时调试视觉部分
  2. 发送策略：状态变化立即发；无变化时按 heartbeat_ms 周期心跳，对端可据此判掉线
  3. 串口异常会自动重连，不会让主循环崩掉
  4. 协议帧 9 字节，见 Protocol；对端按同样格式解析即可

帧格式（大端，无符号）：
    [0]     0xAA
    [1]     0x55
    [2]     CMD     0x10 = 槽位颜色状态
    [3..6]  S1..S4  从左到右 4 个矿石位：0=空(绿) 1=有(红/蓝) 2=未识别
    [7]     SEQ     帧序号 0..255 回绕
    [8]     CRC8    poly=0x07，覆盖 [2]..[7]
"""
import time
from typing import List, Optional

from config import Protocol, RadioConfig

try:
    import serial
    import serial.tools.list_ports
    _HAS_PYSERIAL = True
except ImportError:          # 没装 pyserial 也能跑视觉部分
    serial = None
    _HAS_PYSERIAL = False


class JC24BLink:
    """JC24B 无线串口链路。线程不安全，只在主循环里调用。"""

    def __init__(self, cfg: RadioConfig):
        self.cfg = cfg
        self.ser = None
        self.seq = 0
        self._fail_count = 0
        self._last_send_ms = 0.0
        self._last_payload: Optional[bytes] = None
        self.sent_count = 0
        self.last_error: Optional[str] = None
        self._last_warn_ts = 0.0

    # ------------------------------------------------------------ 生命周期
    def open(self) -> bool:
        if not self.cfg.enabled:
            print("[RADIO] dry-run 模式：不占用串口，仅打印将要发送的帧")
            return False
        if not _HAS_PYSERIAL:
            print("[RADIO] 未安装 pyserial，回退 dry-run。安装： pip install pyserial")
            self.cfg.enabled = False
            return False
        try:
            self.ser = serial.Serial(
                port=self.cfg.port,
                baudrate=self.cfg.baudrate,
                bytesize=self.cfg.bytesize,
                parity=self.cfg.parity,
                stopbits=self.cfg.stopbits,
                timeout=self.cfg.timeout,
                write_timeout=self.cfg.timeout,
            )
            print(f"[RADIO] 已打开 {self.cfg.port} @ {self.cfg.baudrate}")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[RADIO] 打开 {self.cfg.port} 失败：{e}（继续以 dry-run 运行）")
            self.ser = None
            self.cfg.enabled = False
            return False

    def close(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    # ------------------------------------------------------------ 组帧
    @staticmethod
    def frame_with_crc(states: List[int], seq: int) -> bytes:
        """组一帧：AA 55 | CMD | S1 S2 S3 S4 | SEQ | CRC8。SEQ 参与 CRC，让每帧校验值都不同。"""
        body = bytes([
            Protocol.CMD_SLOT_STATE,
            int(states[0]) & 0xFF,
            int(states[1]) & 0xFF,
            int(states[2]) & 0xFF,
            int(states[3]) & 0xFF,
            int(seq) & 0xFF,
        ])
        frame = Protocol.HEAD_BYTES + body + bytes([Protocol.crc8(body)])
        return frame

    # ------------------------------------------------------------ 发送
    def send_slots(self, states: List[int], force: bool = False) -> bool:
        """
        按"变化即发 + 周期心跳"策略发送 4 个槽位状态。
        :param states: 长度为 4 的列表，从左到右，元素取 EMPTY/OCCUPIED/UNKNOWN
        :param force:  True 则忽略策略立即发送
        :return: 是否实际写入（dry-run 下返回 False）
        """
        if len(states) != 4:
            raise ValueError("states 必须正好 4 个槽位（从左到右）")

        now_ms = time.time() * 1000.0
        payload = bytes(states)
        changed = (payload != self._last_payload)
        due = (now_ms - self._last_send_ms) >= self.cfg.heartbeat_ms
        if not force and not changed and not due:
            return False

        self.seq = (self.seq + 1) & 0xFF
        frame = self.frame_with_crc(states, self.seq)
        if self.cfg.append_crlf:
            frame += b"\r\n"

        ok = self._write(frame)
        if ok or not self.cfg.enabled:
            self._last_payload = payload
            self._last_send_ms = now_ms
        return ok

    def _write(self, frame: bytes) -> bool:
        hexs = " ".join(f"{b:02X}" for b in frame)

        if not self.cfg.enabled:
            return False                       # dry-run：静默丢弃
        if self.ser is None or not self.ser.is_open:
            # 串口断开时仍会按策略尝试发送（对端恢复后能立刻收到最新状态），
            # 但告警限流到 1 秒一次，避免刷屏。
            if time.time() - self._last_warn_ts > 1.0:
                self._last_warn_ts = time.time()
                print(f"[RADIO][未连接] {hexs}")
            return False

        try:
            self.ser.write(frame)
            self.ser.flush()
            self.sent_count += 1
            self._fail_count = 0
            return True
        except Exception as e:
            self.last_error = str(e)
            self._fail_count += 1
            print(f"[RADIO] 发送失败({self._fail_count})：{e} | 帧={hexs}")
            if self._fail_count >= self.cfg.reconnect_after_fail:
                self._reconnect()
            return False

    def _reconnect(self):
        print(f"[RADIO] 尝试重连 {self.cfg.port} ...")
        self.close()
        self._fail_count = 0
        self.open()

    # ------------------------------------------------------------ 工具
    @staticmethod
    def list_ports():
        if not _HAS_PYSERIAL:
            print("[RADIO] 未安装 pyserial，无法枚举串口")
            return []
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            print(f"  {p.device}  -  {p.description}")
        return [p.device for p in ports]


class DummyLink(JC24BLink):
    """测试 / 联调用：不碰真实串口，只记录发出的帧，便于验证发送策略。"""

    def __init__(self, cfg: RadioConfig = None):
        super().__init__(cfg or RadioConfig(enabled=True, heartbeat_ms=300))
        self.calls = 0
        self.frames: List[bytes] = []

    def open(self) -> bool:
        print("[RADIO][dummy] 虚拟链路就绪")
        return True

    def is_open(self) -> bool:
        return True

    def _write(self, frame: bytes) -> bool:
        self.calls += 1
        self.frames.append(frame)
        print("[RADIO][dummy] " + " ".join(f"{b:02X}" for b in frame))
        return True


if __name__ == "__main__":
    # 自检：验证组帧与 CRC
    link = JC24BLink(RadioConfig(enabled=False))
    for st in ([0, 1, 0, 1], [1, 1, 0, 0], [2, 2, 2, 2]):
        for seq in (0, 1, 200):
            f = link.frame_with_crc(st, seq)
            assert len(f) == Protocol.FRAME_LEN, f
            assert f[0] == 0xAA and f[1] == 0x55
            assert f[-1] == Protocol.crc8(f[2:-1])
            print(" ".join(f"{b:02X}" for b in f))
    print("协议自检通过")
    JC24BLink.list_ports()
