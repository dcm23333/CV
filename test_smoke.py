# -*- coding: utf-8 -*-
"""
Smoke tests (no camera needed):
  - locate 4 boards, lock, order them left -> right
  - green = EMPTY, red/blue = OCCUPIED
  - 5s stability check (rule 3.4.2)
  - JC24B frame building & send policy
  - regression on a real captured frame (test_data/)

Run: python test_smoke.py
"""
import os
import time

import cv2
import numpy as np

from config import DetectConfig, EMPTY, OCCUPIED, UNKNOWN, STATE_NAME
from detector import SlotTracker
from radio import DummyLink, JC24BLink, RadioConfig


def make_scene(w=640, h=360, pattern=("green", "red", "blue", "green")):
    """Draw 4 boards (100px squares) on a gray court, left to right by pattern."""
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.rectangle(img, (0, h - 60), (w, h), (60, 60, 60), -1)
    bgr = {"green": (60, 200, 60), "red": (50, 50, 230), "blue": (220, 120, 40)}
    size = 100
    ys = h // 2
    xs = [int(w * f) for f in (0.15, 0.38, 0.61, 0.84)]
    for x, cname in zip(xs, pattern):
        cv2.rectangle(img, (x - size // 2, ys - size // 2),
                      (x + size // 2, ys + size // 2), bgr[cname], -1)
        cv2.rectangle(img, (x - size // 2, ys - size // 2),
                      (x + size // 2, ys + size // 2), (40, 40, 40), 2)
    return img


def run_case(pattern, expect, stable_seconds=5.0, frames=200, dt=1 / 30.0, frame=None):
    """frames*dt must exceed stable_seconds, otherwise the stability check can never pass."""
    cfg = DetectConfig(stable_seconds=stable_seconds, min_area=500, max_area=200000)
    tr = SlotTracker(cfg)
    if frame is None:
        frame = make_scene(pattern=pattern)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)

    t = 0.0
    for i in range(frames):
        t += dt
        tr.update(hsv, t)

    got = tr.states()
    print(f"  expect = {[STATE_NAME[s] for s in expect]}")
    print(f"  got    = {[STATE_NAME[s] for s in got]}   "
          f"mode={'LOCKED' if tr.mode == 1 else 'SEARCH'}")
    print(f"  cov    = {[f'{s.raw_coverage:.0%}' for s in tr.slots]}")
    print(f"  stable = {[f'{s.stable_time:.1f}s' for s in tr.slots]} "
          f"all_stable={tr.all_stable()}")

    f = JC24BLink.frame_with_crc(got, 7)
    print(f"  frame  = {' '.join(f'{b:02X}' for b in f)}")

    assert tr.mode == SlotTracker.MODE_LOCK, "failed to lock 4 boards"
    assert got == expect, f"states mismatch: expect {expect}, got {got}"
    assert tr.all_stable(), f"not stable for {stable_seconds}s, min={tr.min_stable_time():.1f}s"


def run_missing_case():
    """Only 2 boards on the court: must NOT lock, states must stay UNKNOWN."""
    cfg = DetectConfig(min_area=500, max_area=200000)
    tr = SlotTracker(cfg)
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (100, 130), (200, 230), (60, 200, 60), -1)
    cv2.rectangle(frame, (400, 130), (500, 230), (50, 50, 230), -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for i in range(20):
        tr.update(hsv, i / 30.0)
    print(f"  2-board scene: mode={'LOCKED' if tr.mode == 1 else 'SEARCH'} "
          f"slots={len(tr.slots)} (expect SEARCH / 0)")
    assert tr.mode == SlotTracker.MODE_SEARCH and len(tr.slots) == 0


def run_search_clears_slots():
    """After losing the boards, going back to SEARCH must clear stale slots
    (otherwise the UI keeps showing states from dead boxes)."""
    cfg = DetectConfig(min_area=500, max_area=200000, miss_frames=5)
    tr = SlotTracker(cfg)
    frame = make_scene(pattern=("green", "red", "blue", "green"))
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    for i in range(30):
        tr.update(hsv, i / 30.0)
    assert tr.mode == SlotTracker.MODE_LOCK and len(tr.slots) == 4

    blank = np.full(frame.shape, 120, dtype=np.uint8)     # boards disappear
    hsv2 = cv2.cvtColor(blank, cv2.COLOR_BGR2HSV)
    for i in range(10):
        tr.update(hsv2, 1.0 + i / 30.0)
    print(f"  boards removed: mode={'LOCKED' if tr.mode == 1 else 'SEARCH'} "
          f"slots={len(tr.slots)} (expect SEARCH / 0)")
    assert tr.mode == SlotTracker.MODE_SEARCH and len(tr.slots) == 0
    assert tr.states() == [UNKNOWN] * 4


def run_radio_policy():
    """change-triggered send + heartbeat policy."""
    link = DummyLink(RadioConfig(enabled=True, heartbeat_ms=300))
    link.send_slots([EMPTY, OCCUPIED, EMPTY, OCCUPIED])   # changed   -> send
    n1 = link.calls
    link.send_slots([EMPTY, OCCUPIED, EMPTY, OCCUPIED])   # unchanged -> skip
    n2 = link.calls
    time.sleep(0.35)
    link.send_slots([EMPTY, OCCUPIED, EMPTY, OCCUPIED])   # heartbeat -> send
    n3 = link.calls
    link.send_slots([EMPTY, OCCUPIED, EMPTY, UNKNOWN])    # changed   -> send
    n4 = link.calls
    print(f"  radio: calls after change/no-change/heartbeat/change = {n1}/{n2}/{n3}/{n4}")
    assert (n1, n2, n3, n4) == (1, 1, 2, 3), "send policy mismatch"
    assert len(link.frame_with_crc([0, 1, 0, 1], 0)) == 9
    assert len(link.frames[0]) == 9


def run_real_frame_regression():
    """Regression on a real captured frame (the one that used to fail:
    light-green boards were missed because their saturation is only ~49)."""
    path = os.path.join("test_data", "scene_green_blue_blue_green.png")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return
    frame = cv2.imread(path)
    run_case(("green", "blue", "blue", "green"),
             [EMPTY, OCCUPIED, OCCUPIED, EMPTY], frame=frame)


if __name__ == "__main__":
    print("=== case 1: green red blue green -> EMPTY OCC OCC EMPTY ===")
    run_case(("green", "red", "blue", "green"), [EMPTY, OCCUPIED, OCCUPIED, EMPTY])
    print("=== case 2: all green ===")
    run_case(("green", "green", "green", "green"), [EMPTY] * 4)
    print("=== case 3: red green green blue (typical 2-mineral layout) ===")
    run_case(("red", "green", "green", "blue"), [OCCUPIED, EMPTY, EMPTY, OCCUPIED])
    print("=== case 4: only 2 boards -> no lock ===")
    run_missing_case()
    print("=== case 5: stale slots cleared when returning to SEARCH ===")
    run_search_clears_slots()
    print("=== case 6: radio send policy ===")
    run_radio_policy()
    print("=== case 7: real-frame regression (light green boards) ===")
    run_real_frame_regression()
    print("\nAll smoke tests passed.")
