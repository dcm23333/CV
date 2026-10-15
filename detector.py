# -*- coding: utf-8 -*-
"""
颜色识别核心。

职责：
  1. 在画面中定位 4 个颜色识别板（全局搜索 -> 锁定跟踪 -> 丢失重搜）
  2. 按摄像头视角从左到右排序，一一对应 4 个矿石位
  3. 对每个色板判定颜色：绿色 -> 空位；红色/蓝色 -> 有矿石
  4. 时间域众数滤波 + 5 秒稳定判定（规则 3.4.2）

对外只需要两个对象：SlotTracker（每帧 update）和它的 .slots。
"""
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    EMPTY, OCCUPIED, UNKNOWN,
    EMPTY_COLORS, OCCUPIED_COLORS,
    DetectConfig, HSV_RANGES,
)


# ================================================================ 单个槽位
class Slot:
    """一个矿石位：几何位置 + 状态历史 + 稳定计时。"""

    def __init__(self, slot_id: int, box: Tuple[int, int, int, int], vote_window: int,
                 stable_seconds: float):
        self.id = slot_id                  # 0..3，从左到右
        self.box = box                     # (x, y, w, h)，基于检测分辨率
        self._vote_window = vote_window
        self._stable_seconds = stable_seconds

        self.history: deque = deque(maxlen=vote_window)
        self.raw_state = UNKNOWN           # 本帧原始判定
        self.raw_color = None              # 本帧命中的颜色名
        self.raw_coverage = 0.0            # 本帧最大颜色覆盖率
        self.state = UNKNOWN               # 滤波后输出状态
        self.color = None                  # 滤波后输出颜色
        self._last_state = UNKNOWN
        self._stable_since: Optional[float] = None
        self.stable_time = 0.0             # 当前状态已连续保持的秒数
        self.lost_frames = 0               # 连续未检测到有效色板的帧数

    # -------------------------------------------------- 单帧更新
    def push(self, state: int, color: Optional[str], coverage: float, now: float):
        self.raw_state, self.raw_color, self.raw_coverage = state, color, coverage
        self.history.append(state)
        self.lost_frames = 0 if state != UNKNOWN else self.lost_frames + 1

        if len(self.history) >= max(3, self._vote_window // 2):
            # 众数投票；平票时偏向"已知"状态，避免 UNKNOWN 卡死
            cnt = Counter(self.history)
            top = cnt.most_common()
            best, best_n = top[0]
            if len(top) > 1 and top[1][1] == best_n:
                for s, _ in top:
                    if s != UNKNOWN:
                        best = s
                        break
            new_state = best
            color_cnt = Counter(h for h in [self.raw_color] if h)
            if color_cnt:
                self.color = color_cnt.most_common(1)[0][0]
            elif new_state == UNKNOWN:
                self.color = None
        else:
            new_state = UNKNOWN

        if new_state != self.state:
            self.state = new_state
            self._stable_since = now
            self.stable_time = 0.0
        elif self._stable_since is not None:
            self.stable_time = now - self._stable_since
        else:
            self._stable_since = now

    # -------------------------------------------------- 稳定判定
    @property
    def is_stable(self) -> bool:
        """规则 3.4.2：该槽位持续 5 秒无丢失/误识别。"""
        return self.state != UNKNOWN and self.stable_time >= self._stable_seconds

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.box
        return (int(x + w / 2), int(y + h / 2))

    def __repr__(self):
        return f"<Slot#{self.id} {self.box} state={self.state} stable={self.stable_time:.1f}s>"


# ================================================================ 颜色工具
def build_masks(hsv: np.ndarray, cfg: DetectConfig) -> Dict[str, np.ndarray]:
    """生成各颜色的二值掩膜，并做形态学清洗。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    masks = {}
    for name, ranges in HSV_RANGES.items():
        m = None
        for lo, hi in ranges:
            cur = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            m = cur if m is None else cv2.bitwise_or(m, cur)
        if m is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        masks[name] = m
    return masks


def classify_region(masks: Dict[str, np.ndarray], box: Tuple[int, int, int, int],
                    cfg: DetectConfig) -> Tuple[int, Optional[str], float]:
    """在给定 ROI 内统计各颜色占比，返回 (state, color_name, coverage)。"""
    x, y, w, h = box
    H, W = next(iter(masks.values())).shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return UNKNOWN, None, 0.0

    area = float((x1 - x0) * (y1 - y0))
    best_color, best_cov = None, 0.0
    for name, m in masks.items():
        cov = float(cv2.countNonZero(m[y0:y1, x0:x1])) / area
        if cov > best_cov:
            best_color, best_cov = name, cov

    if best_color is None or best_cov < cfg.min_coverage:
        return UNKNOWN, None, best_cov

    if best_color in EMPTY_COLORS:
        return EMPTY, best_color, best_cov
    if best_color in OCCUPIED_COLORS:
        return OCCUPIED, best_color, best_cov
    return UNKNOWN, best_color, best_cov


# ================================================================ 跟踪器
class SlotTracker:
    """定位 4 个色板并持续跟踪。三种工作模式：auto(默认) / manual / 外部注入。"""

    MODE_SEARCH, MODE_LOCK = 0, 1

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.slots: List[Slot] = []
        self.mode = self.MODE_SEARCH
        self._lock_hits = 0
        self._miss = 0
        self._pending: List[Tuple[int, int, int, int]] = []
        self.last_candidates = 0   # 上一帧找到的候选框数量，用于排查"锁不上"的原因

        if cfg.manual_mode and cfg.roi_file:
            self._load_manual_roi(cfg.roi_file)

    # -------------------------------------------------- 手动 ROI
    def _load_manual_roi(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        boxes = [tuple(map(int, b)) for b in data["slots"]]
        boxes.sort(key=lambda b: b[0])           # 从左到右
        assert len(boxes) == self.cfg.slot_count, "手动 ROI 数量必须等于 slot_count"
        self.slots = [Slot(i, b, self.cfg.vote_window, self.cfg.stable_seconds)
                      for i, b in enumerate(boxes)]
        self.mode = self.MODE_LOCK

    def set_manual_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        boxes = sorted(boxes, key=lambda b: b[0])
        self.slots = [Slot(i, b, self.cfg.vote_window, self.cfg.stable_seconds)
                      for i, b in enumerate(boxes)]
        self.mode = self.MODE_LOCK

    # -------------------------------------------------- 候选轮廓
    def _find_candidates(self, masks: Dict[str, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        all_mask = None
        for m in masks.values():
            all_mask = m if all_mask is None else cv2.bitwise_or(all_mask, m)
        if all_mask is None:
            return []

        contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            ar = w / float(h)
            if not (self.cfg.aspect_range[0] <= ar <= self.cfg.aspect_range[1]):
                continue
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            if ha > 0 and area / ha < self.cfg.min_solidity:
                continue
            boxes.append((x, y, w, h))
        return boxes

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]],
                     gap: int = 25) -> List[Tuple[int, int, int, int]]:
        """合并水平方向邻近的框（同一色板被 HSV 分段切碎的情况）。"""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [list(boxes[0])]
        for x, y, w, h in boxes[1:]:
            px, py, pw, ph = merged[-1]
            if x <= px + pw + gap and not (y > py + ph or py > y + h):
                x2 = max(px + pw, x + w)
                y2 = max(py + ph, y + h)
                merged[-1] = [min(px, x), min(py, y), x2 - min(px, x), y2 - min(py, y)]
            else:
                merged.append([x, y, w, h])
        return [tuple(map(int, b)) for b in merged]

    def _pick_four(self, boxes: List[Tuple[int, int, int, int]]
                   ) -> Optional[List[Tuple[int, int, int, int]]]:
        """从候选框中挑选 4 个最像色板的：面积接近 + 垂直位置接近 + 水平分布均匀。"""
        boxes = self._merge_boxes(boxes)
        if len(boxes) < self.cfg.slot_count:
            return None
        boxes = sorted(boxes, key=lambda b: b[0])
        cands = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:12]

        best, best_score = None, -1e18
        n = self.cfg.slot_count
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                for k in range(j + 1, len(cands)):
                    for l in range(k + 1, len(cands)):
                        grp = [cands[i], cands[j], cands[k], cands[l]]
                        areas = [b[2] * b[3] for b in grp]
                        cys = [b[1] + b[3] / 2 for b in grp]
                        cxs = [b[0] + b[2] / 2 for b in grp]
                        area_cv = np.std(areas) / (np.mean(areas) + 1e-6)     # 面积一致性
                        y_cv = np.std(cys) / (np.mean(cys) + 1e-6)            # 同一水平线
                        gaps = np.diff(sorted(cxs))
                        gap_cv = np.std(gaps) / (np.mean(gaps) + 1e-6)        # 间距均匀
                        score = -(2.0 * area_cv + 1.5 * y_cv + 1.0 * gap_cv) + \
                                np.mean(areas) / 1e5
                        if score > best_score:
                            best_score, best = score, grp
        if best is None:
            return None
        return sorted(best, key=lambda b: b[0])   # 从左到右

    # -------------------------------------------------- 主更新
    def update(self, hsv: np.ndarray, now: float) -> List[Slot]:
        masks = build_masks(hsv, self.cfg)

        if self.mode == self.MODE_SEARCH or not self.slots:
            raw = self._find_candidates(masks)
            self.last_candidates = len(self._merge_boxes(raw))
            boxes = self._pick_four(raw)
            if boxes:
                self._pending.append(boxes)
                if len(self._pending) > self.cfg.lock_frames:
                    self._pending.pop(0)
                if len(self._pending) >= self.cfg.lock_frames:
                    # 连续多帧的 4 个框中心都足够接近才锁定
                    ref = self._pending[0]
                    ok = all(
                        all(abs(a[0] + a[2] / 2 - b[0] - b[2] / 2) < self.cfg.max_drift and
                            abs(a[1] + a[3] / 2 - b[1] - b[3] / 2) < self.cfg.max_drift
                            for a, b in zip(ref, p))
                        for p in self._pending[1:]
                    )
                    if ok:
                        self.slots = [Slot(i, b, self.cfg.vote_window, self.cfg.stable_seconds)
                                      for i, b in enumerate(boxes)]
                        self.mode = self.MODE_LOCK
                        self._miss = 0
                        self._pending.clear()

        if self.mode == self.MODE_LOCK and self.slots:
            # 手动 ROI 模式：位置是人工框定的，不再做跟踪更新，避免被误检带跑
            if self.cfg.manual_mode:
                self._miss = 0
                for s in self.slots:
                    box = self._expand(s.box, self.cfg.roi_scale, hsv.shape)
                    st, color, cov = classify_region(masks, box, self.cfg)
                    s.push(st, color, cov, now)
                return self.slots

            # 锁定态：在上一帧位置附近重新搜索，实现小幅漂移跟踪
            boxes = self._find_candidates(masks)
            self.last_candidates = len(self._merge_boxes(boxes))
            if boxes:
                new_boxes = []
                for s in self.slots:
                    cx, cy = s.center
                    near = [b for b in boxes
                            if abs(b[0] + b[2] / 2 - cx) < self.cfg.max_drift and
                               abs(b[1] + b[3] / 2 - cy) < self.cfg.max_drift]
                    if near:
                        near.sort(key=lambda b: (b[0] + b[2] / 2 - cx) ** 2 +
                                                (b[1] + b[3] / 2 - cy) ** 2)
                        new_boxes.append(near[0])
                    else:
                        new_boxes.append(s.box)
                # 保持从左到右顺序，防止跟踪时序号错乱
                order = np.argsort([b[0] for b in new_boxes])
                for i, idx in enumerate(order):
                    self.slots[i].box = new_boxes[idx]
                self._miss = 0 if len(boxes) >= self.cfg.slot_count else self._miss + 1
            else:
                self._miss += 1

            if self._miss > self.cfg.miss_frames and not self.cfg.manual_mode:
                # 重新全局搜索时必须丢弃旧槽位，否则会用失效的框继续输出状态，
                # 表现为"SEARCHING 但顶部卡片还在显示上一次的结果"。
                self.mode = self.MODE_SEARCH
                self.slots = []
                self._pending.clear()
                self._miss = 0

        # 颜色判定 + 时间滤波
        for s in self.slots:
            box = self._expand(s.box, self.cfg.roi_scale, hsv.shape)
            st, color, cov = classify_region(masks, box, self.cfg)
            s.push(st, color, cov, now)

        return self.slots

    @staticmethod
    def _expand(box, scale, shape):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        nw, nh = w * scale, h * scale
        H, W = shape[:2]
        nx = int(max(0, min(W - 1, cx - nw / 2)))
        ny = int(max(0, min(H - 1, cy - nh / 2)))
        return (nx, ny, int(min(W - nx, nw)), int(min(H - ny, nh)))

    # -------------------------------------------------- 结果导出
    def states(self) -> List[int]:
        if not self.slots:
            return [UNKNOWN] * self.cfg.slot_count
        return [s.state for s in self.slots]

    def all_stable(self) -> bool:
        return len(self.slots) == self.cfg.slot_count and all(s.is_stable for s in self.slots)

    def min_stable_time(self) -> float:
        if not self.slots:
            return 0.0
        return min(s.stable_time for s in self.slots)

    def reset_lock(self):
        """丢弃当前锁定，重新全局搜索。"""
        self.slots = []
        self.mode = self.MODE_SEARCH
        self._pending.clear()
        self._miss = 0
