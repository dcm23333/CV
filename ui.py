# -*- coding: utf-8 -*-
"""
绘制层：把 4 个槽位的识别结果画到画面上。

规则 3.4.2 要求 UI 明确显示四个目标对应的识别结果，因此这里给出：
  - 画面上每个色板的包围框、序号(1..4，从左到右)、颜色名
  - 顶部 4 张状态卡片：EMPTY(绿) / OCCUPIED(红蓝) / UNKNOWN(灰)
  - FPS、跟踪模式、稳定倒计时、整体判定结果

注意：cv2.putText 不支持中文，画面上所有文字一律用英文，否则会渲染成 ????????
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import EMPTY, OCCUPIED, UNKNOWN, STATE_BGR, STATE_NAME
from detector import Slot, SlotTracker

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, scale=0.6, color=(255, 255, 255), thick=2, bg=None):
    """带描边的文字，保证在任何背景上都看得清。"""
    if bg is not None:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base + 3), bg, -1)
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame: np.ndarray, tracker: SlotTracker, fps: float, scale: float,
         recognized: bool, radio_info: str = "") -> np.ndarray:
    """
    :param frame:   原始分辨率画面（绘制用）
    :param tracker: SlotTracker 实例
    :param scale:   检测分辨率 -> 原始分辨率 的放大系数
    :param recognized: 是否满足"4 个槽位全部稳定 >= 5 秒"
    """
    vis = frame.copy()
    H, W = vis.shape[:2]

    # ---------------- 画面上的色板框 ----------------
    for s in tracker.slots:
        x, y, w, h = s.box
        x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
        c = STATE_BGR[s.state]
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 3)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, c, -1)
        label = f"#{s.id + 1} {STATE_NAME[s.state]}"
        if s.color:
            label += f"/{s.color}"
        _put(vis, label, (x, y - 8), 0.55, c, 2, bg=(20, 20, 20))
        _put(vis, f"{s.raw_coverage:.0%}", (x, y + h + 18), 0.45, (190, 190, 190), 1,
             bg=(20, 20, 20))
        # 稳定进度条
        bar_w = int(w * min(1.0, s.stable_time / tracker.cfg.stable_seconds))
        cv2.rectangle(vis, (x, y + h + 2), (x + w, y + h + 6), (70, 70, 70), -1)
        cv2.rectangle(vis, (x, y + h + 2), (x + bar_w, y + h + 6), c, -1)

    # ---------------- 顶部 4 张状态卡片 ----------------
    states: List[int] = tracker.states()
    n = tracker.cfg.slot_count
    card_w = min(150, W // (n + 1))
    card_h = 74
    x0 = (W - card_w * n - 10 * (n - 1)) // 2
    y0 = 12
    for i in range(n):
        st = states[i] if i < len(states) else UNKNOWN
        c = STATE_BGR[st]
        x = x0 + i * (card_w + 10)
        overlay = vis.copy()
        cv2.rectangle(overlay, (x, y0), (x + card_w, y0 + card_h), c, -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        cv2.rectangle(vis, (x, y0), (x + card_w, y0 + card_h), c, 2)
        _put(vis, f"POS {i + 1}", (x + 10, y0 + 26), 0.55, (255, 255, 255), 2)
        _put(vis, STATE_NAME[st], (x + 10, y0 + 54), 0.55, (255, 255, 255), 2)
    _put(vis, "<- LEFT", (x0 - 78, y0 + 40), 0.45, (200, 200, 200), 1)
    _put(vis, "RIGHT ->", (x0 + card_w * n + 10 * (n - 1) + 8, y0 + 40), 0.45,
         (200, 200, 200), 1)

    # ---------------- 信息栏 ----------------
    if tracker.cfg.manual_mode:
        mode_txt = "MANUAL-ROI"
        mode_c = (0, 220, 255)
    elif tracker.mode == SlotTracker.MODE_LOCK:
        mode_txt, mode_c = "LOCKED", (80, 220, 120)
    else:
        mode_txt, mode_c = "SEARCHING...", (80, 180, 250)
    fps_c = (80, 220, 120) if fps >= 30 else (60, 60, 235)   # 规则要求 >= 30fps
    info_y = y0 + card_h + 26
    _put(vis, f"FPS {fps:5.1f}", (12, info_y), 0.6, fps_c, 2, bg=(20, 20, 20))
    _put(vis, mode_txt, (130, info_y), 0.6, mode_c, 2, bg=(20, 20, 20))
    if tracker.slots:
        _put(vis, f"STABLE {tracker.min_stable_time():.1f}/{tracker.cfg.stable_seconds:.0f}s",
             (270, info_y), 0.6, (235, 235, 235), 2, bg=(20, 20, 20))
    _put(vis, radio_info, (12, info_y + 26), 0.5, (180, 200, 255), 1, bg=(20, 20, 20))

    # ---------------- 判定结果横幅 ----------------
    if recognized:
        txt = "RECOGNIZED  (4/4 stable >= 5s)  +10"
        (tw, th), base = cv2.getTextSize(txt, FONT, 0.8, 2)
        bx = (W - tw) // 2
        by = H - 40
        cv2.rectangle(vis, (bx - 12, by - th - 10), (bx + tw + 12, by + 10), (30, 150, 60), -1)
        _put(vis, txt, (bx, by), 0.8, (255, 255, 255), 2)

    _put(vis, "q:quit  r:reset  t:tune  s:save  m:manual-ROI  l:dump-ROI",
         (12, H - 12), 0.45, (150, 150, 150), 1)
    return vis


# ================================================================ 鼠标框选 ROI
class RoiPicker:
    """
    手动标定：在画面上依次拖 4 个框（必须从左到右），框定 4 个色板位置。
    背景杂乱导致自动搜索锁不上时，机位固定后直接用手动 ROI 最稳。
    """

    def __init__(self, win: str, count: int = 4):
        self.win = win
        self.count = count
        self.boxes: List[Tuple[int, int, int, int]] = []
        self.dragging = False
        self.start: Optional[Tuple[int, int]] = None
        self.cur: Optional[Tuple[int, int]] = None
        self.done = False

    def attach(self):
        cv2.setMouseCallback(self.win, self._cb)

    def detach(self):
        cv2.setMouseCallback(self.win, lambda *a: None)

    def _cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging, self.start, self.cur = True, (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self.start
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if box[2] >= 8 and box[3] >= 8:
                self.boxes.append(box)
                if len(self.boxes) >= self.count:
                    self.done = True

    def to_detect(self, scale: float) -> List[Tuple[int, int, int, int]]:
        """窗口坐标(原始分辨率) -> 检测分辨率坐标。"""
        out = []
        for (x, y, w, h) in self.boxes:
            out.append((int(x / scale), int(y / scale), max(4, int(w / scale)),
                        max(4, int(h / scale))))
        return out

    def draw(self, vis: np.ndarray):
        for i, (x, y, w, h) in enumerate(self.boxes):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), 2)
            _put(vis, str(i + 1), (x + 4, y + 22), 0.6, (0, 220, 255), 2, bg=(20, 20, 20))
        if self.dragging and self.start and self.cur:
            cv2.rectangle(vis, self.start, self.cur, (0, 255, 255), 1)
        tip = f"MANUAL ROI: drag {len(self.boxes)}/{self.count} boxes, LEFT to RIGHT"
        _put(vis, tip, (12, 32), 0.7, (0, 230, 255), 2, bg=(20, 20, 20))
        _put(vis, "keep the same camera position afterwards", (12, 56), 0.45,
             (180, 220, 255), 1, bg=(20, 20, 20))


# ================================================================ HSV 调参窗口
class Tuner:
    """
    运行时 HSV 阈值调参窗口，现场换灯光时直接拖滑块校准。

    绿色单独用较低的 Smin：浅绿塑料板饱和度常常只有 45~50，
    若和红蓝共用同一个较高的 Smin，绿色会整块漏检（实测踩过这个坑）。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.win = "HSV Tuner"
        self._built = False

    def build(self):
        if self._built:
            return
        cv2.namedWindow(self.win)
        cv2.createTrackbar("G Hmin", self.win, 40, 180, lambda v: None)
        cv2.createTrackbar("G Hmax", self.win, 85, 180, lambda v: None)
        cv2.createTrackbar("G Smin", self.win, 30, 255, lambda v: None)
        cv2.createTrackbar("R Smin", self.win, 60, 255, lambda v: None)
        cv2.createTrackbar("B Smin", self.win, 65, 255, lambda v: None)
        cv2.createTrackbar("R Hmax1", self.win, 12, 90, lambda v: None)
        cv2.createTrackbar("R Hmin2", self.win, 168, 180, lambda v: None)
        cv2.createTrackbar("B Hmin", self.win, 88, 180, lambda v: None)
        cv2.createTrackbar("B Hmax", self.win, 132, 180, lambda v: None)
        cv2.createTrackbar("Vmin(all)", self.win, 45, 255, lambda v: None)
        cv2.createTrackbar("minArea/100", self.win, 3, 200, lambda v: None)
        cv2.createTrackbar("minCov%", self.win, 12, 100, lambda v: None)
        self._built = True

    def apply(self):
        """把滑块值写回全局 HSV_RANGES / DetectConfig。"""
        import config
        g = lambda n: cv2.getTrackbarPos(n, self.win)
        vmin = g("Vmin(all)")
        config.HSV_RANGES["green"] = [((g("G Hmin"), g("G Smin"), vmin),
                                       (g("G Hmax"), 255, 255))]
        config.HSV_RANGES["red"] = [
            ((0, g("R Smin"), vmin), (g("R Hmax1"), 255, 255)),
            ((g("R Hmin2"), g("R Smin"), vmin), (180, 255, 255)),
        ]
        config.HSV_RANGES["blue"] = [((g("B Hmin"), g("B Smin"), vmin),
                                      (g("B Hmax"), 255, 255))]
        self.cfg.min_area = max(50, g("minArea/100") * 100)
        self.cfg.min_coverage = max(0.01, g("minCov%") / 100.0)

    def update(self):
        self.build()
        self.apply()
