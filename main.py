# -*- coding: utf-8 -*-
"""
AR 升级区颜色识别主程序。

功能（对应规则手册 2.2.5 / 3.4.2）：
  - 识别 AR 升级区 4 个矿石位，按摄像头视角从左到右一一对应
  - 绿色 -> 空位(EMPTY)；红色/蓝色 -> 有矿石(OCCUPIED)
  - UI 明确显示 4 个结果，处理帧率 >= 30fps，稳定 5 秒判定识别成功
  - 通过 JC24B 无线串口把 4 个槽位状态发给 AR 主控

常用命令：
  python main.py --camera 0                          # 默认 dry-run，不发串口
  python main.py --camera 0 --serial COM3            # 打开 JC24B 发送
  python main.py --video test.mp4                    # 用录像离线调试
  python main.py --camera 0 --tune                   # 打开 HSV 调参窗口
  python main.py --list-ports                        # 枚举可用串口
  python main.py --camera 0 --no-ui                  # 无界面，只跑识别+发送
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from config import (CameraConfig, DetectConfig, RadioConfig, EMPTY, OCCUPIED, UNKNOWN,
                    STATE_NAME)
from detector import SlotTracker
from radio import JC24BLink
from ui import RoiPicker, Tuner, draw

WIN = "AR Slot CV"


def parse_args():
    p = argparse.ArgumentParser(
        description="ENTERPRIZE RM2027 - AR upgrade zone color detection")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--video", type=str, default=None,
                   help="video file instead of camera (offline debug)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--detect-width", type=int, default=640,
                   help="width of the downscaled frame used for detection (smaller=faster)")
    p.add_argument("--fourcc", type=str, default=None, help='e.g. "MJPG"')

    p.add_argument("--serial", type=str, default=None,
                   help="JC24B serial port, e.g. COM3 / /dev/ttyUSB0; omit = dry-run")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--heartbeat", type=int, default=500, help="heartbeat period in ms")
    p.add_argument("--list-ports", action="store_true", help="list serial ports and exit")

    p.add_argument("--roi-file", type=str, default=None,
                   help='manual ROI json: {"slots":[[x,y,w,h],...]} in detect resolution')
    p.add_argument("--min-area", type=int, default=None)
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--stable", type=float, default=5.0,
                   help="seconds a state must hold (rule requires 5s)")

    p.add_argument("--tune", action="store_true", help="open HSV tuner at startup")
    p.add_argument("--no-ui", action="store_true", help="headless, no window")
    p.add_argument("--save-dir", type=str, default="captures",
                   help="where 's' snapshots and ROI dumps go")
    return p.parse_args()


def open_capture(cam: CameraConfig, video: str = None):
    src = video if video else cam.index
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[CAM] cannot open {src}")
        return None
    if video is None:
        if cam.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        cap.set(cv2.CAP_PROP_FPS, cam.fps)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if cam.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, cam.exposure)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] source={src} resolution={w}x{h}")
    return cap


def main():
    args = parse_args()

    if args.list_ports:
        JC24BLink.list_ports()
        return

    cam_cfg = CameraConfig(index=args.camera, width=args.width, height=args.height,
                           fps=args.fps, detect_width=args.detect_width, fourcc=args.fourcc)
    det_cfg = DetectConfig(roi_file=args.roi_file, manual_mode=bool(args.roi_file),
                           stable_seconds=args.stable)
    if args.min_area:
        det_cfg.min_area = args.min_area
    if args.min_coverage:
        det_cfg.min_coverage = args.min_coverage

    radio_cfg = RadioConfig(enabled=bool(args.serial), port=args.serial or "COM3",
                            baudrate=args.baud, heartbeat_ms=args.heartbeat)

    cap = open_capture(cam_cfg, args.video)
    if cap is None:
        sys.exit(1)

    tracker = SlotTracker(det_cfg)
    link = JC24BLink(radio_cfg)
    link.open()
    tuner = Tuner(det_cfg) if (args.tune and not args.no_ui) else None
    if tuner:
        tuner.build()

    os.makedirs(args.save_dir, exist_ok=True)
    picker = None
    scale = 1.0
    fps = 0.0
    prev_t = time.time()
    t0 = time.time()
    frame_id = 0

    print("[MAIN] started. keys: q=quit r=reset t=tune s=snapshot m=manual-ROI l=dump-ROI")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAM] frame read failed / video ended")
                if args.video:
                    break
                time.sleep(0.02)
                continue

            now = time.time() - t0
            frame_id += 1

            # ---- 降采样做检测，保证 >= 30fps ----
            H, W = frame.shape[:2]
            det_w = cam_cfg.detect_width
            small = cv2.resize(frame, (det_w, int(H * det_w / W)), interpolation=cv2.INTER_AREA)
            scale = W / float(det_w)
            if det_cfg.blur_kernel > 1:
                small = cv2.GaussianBlur(small, (det_cfg.blur_kernel, det_cfg.blur_kernel), 0)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            slots = tracker.update(hsv, now)
            states = tracker.states()
            recognized = tracker.all_stable()

            # ---- 发送给 AR（变化即发 + 心跳）----
            link.send_slots(states)

            # ---- FPS ----
            cur = time.time()
            inst = 1.0 / max(1e-6, cur - prev_t)
            prev_t = cur
            fps = fps * 0.85 + inst * 0.15 if frame_id > 1 else inst

            # ---- 串口状态栏 ----
            if link.is_open():
                rinfo = f"RADIO {radio_cfg.port}@{radio_cfg.baud} tx={link.sent_count} seq={link.seq}"
            elif radio_cfg.enabled:
                rinfo = f"RADIO NOT CONNECTED ({link.last_error})"
            else:
                rinfo = "RADIO dry-run (serial disabled)  |  " + \
                        "/".join(STATE_NAME[s] for s in states)
            rinfo += f"  |  cand={tracker.last_candidates}/{det_cfg.slot_count}"

            if not args.no_ui:
                vis = draw(frame, tracker, fps, scale, recognized, rinfo)
                if picker is not None:
                    picker.draw(vis)
                if tuner:
                    tuner.update()
                cv2.imshow(WIN, vis)
                key = cv2.waitKey(1) & 0xFF

                if picker is not None and picker.done:
                    # 框完 4 个立即切到手动 ROI 模式
                    det_cfg.roi_file = None
                    det_cfg.manual_mode = True
                    tracker.set_manual_boxes(picker.to_detect(scale))
                    picker.detach()
                    picker = None
                    print(f"[MAIN] manual ROI applied: "
                          f"{[list(map(int, s.box)) for s in tracker.slots]}")

                if key == ord("q"):
                    break
                elif key == ord("r"):
                    tracker.reset_lock()
                    det_cfg.manual_mode = bool(args.roi_file)
                    print("[MAIN] lock reset, searching again")
                elif key == ord("s"):
                    path = os.path.join(args.save_dir, f"frame_{int(time.time())}.png")
                    cv2.imwrite(path, frame)
                    print(f"[MAIN] saved {path}")
                elif key == ord("m"):
                    picker = RoiPicker(WIN, det_cfg.slot_count)
                    picker.attach()
                    print("[MAIN] drag 4 boxes from LEFT to RIGHT")
                elif key == ord("l"):
                    boxes = [list(map(int, s.box)) for s in tracker.slots]
                    path = os.path.join(args.save_dir, "roi.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"slots": boxes, "detect_width": det_w}, f,
                                  ensure_ascii=False, indent=2)
                    print(f"[MAIN] dumped ROI to {path} "
                          f"(reuse with --roi-file {path})")
                elif key == ord("t"):
                    tuner = Tuner(det_cfg)
                    tuner.build()
            else:
                if frame_id % 30 == 0:
                    print(f"[{frame_id:6d}] fps={fps:5.1f} states={states} "
                          f"cand={tracker.last_candidates} "
                          f"stable={tracker.min_stable_time():.1f}s "
                          f"recognized={recognized}")

    except KeyboardInterrupt:
        print("\n[MAIN] interrupted by user")
    finally:
        cap.release()
        link.close()
        cv2.destroyAllWindows()
        print(f"[MAIN] done. {frame_id} frames processed, "
              f"{link.sent_count} radio frames sent")


if __name__ == "__main__":
    main()
