"""
Debug script: visualize why players don't get ankle midpoint dots.

For each sampled frame, draws:
  - Green boxes: player detections (YOLO person model)
  - Blue boxes:  pose model detections
  - Cyan circles: left foot keypoint (kp0)
  - Magenta circles: right foot keypoint (kp1)
  - Red dot: assigned ankle midpoint
  - Per-player text: best IoU, keypoint confidences, reason for failure

Usage:
    python scripts/debug_pose_ankles.py --video data/rally/video.mp4
    python scripts/debug_pose_ankles.py --video data/rally/video.mp4 --start 100 --end 200 --step 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from padeldet.player_tracker.player_tracker import PlayerTrackerConfig, _box_iou
from padeldet.video import frame_generator, VideoInfo
from ultralytics import YOLO


POSE_MODEL_DEFAULT = "yolo26m-pose.pt"
PLAYER_MODEL_DEFAULT = "weights/yolov8m.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default=PLAYER_MODEL_DEFAULT)
    p.add_argument("--pose-model", default=POSE_MODEL_DEFAULT)
    p.add_argument("--output", default=None, help="Output video (default: <video_dir>/debug_pose_ankles.mp4)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=5, help="Sample every N frames (default: 5)")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.3, help="Pose IoU match threshold")
    p.add_argument("--kp-conf", type=float, default=0.1, help="Min keypoint confidence")
    p.add_argument("--lf-kp", type=int, default=15, help="Left ankle keypoint index (default: 15 COCO)")
    p.add_argument("--rf-kp", type=int, default=16, help="Right ankle keypoint index (default: 16 COCO)")
    args = p.parse_args()
    if args.output is None:
        args.output = str(Path(args.video).parent / "debug_pose_ankles.mp4")
    return args


def draw_text_bg(img, text, pos, font_scale=0.45, thickness=1, color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 2


def analyze_frame(frame, player_result, pose_result, iou_threshold, kp_conf_threshold, lf_kp=15, rf_kp=16):
    """Return annotated frame with full diagnostic overlay."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # --- Draw all pose detections (blue boxes) ---
    pose_boxes = np.empty((0, 4))
    kp_xy = None
    kp_conf_arr = None

    if pose_result is not None and pose_result.boxes is not None and len(pose_result.boxes):
        pose_boxes = pose_result.boxes.xyxy.cpu().numpy()
        if pose_result.keypoints is not None and pose_result.keypoints.xy is not None:
            kp_xy = pose_result.keypoints.xy.cpu().numpy()       # (N, K, 2)
            if pose_result.keypoints.conf is not None:
                kp_conf_arr = pose_result.keypoints.conf.cpu().numpy()  # (N, K)

        for i, (px1, py1, px2, py2) in enumerate(pose_boxes):
            cv2.rectangle(vis, (int(px1), int(py1)), (int(px2), int(py2)), (255, 180, 0), 1)

            if kp_xy is not None and i < len(kp_xy):
                for kp_idx, kp_color, kp_name in [
                    (lf_kp, (255, 255, 0), "LF"),
                    (rf_kp, (255, 0, 255), "RF"),
                ]:
                    kx, ky = kp_xy[i, kp_idx]
                    if kx == 0.0 and ky == 0.0:
                        continue
                    kp_c = float(kp_conf_arr[i, kp_idx]) if kp_conf_arr is not None else -1.0
                    cv2.circle(vis, (int(kx), int(ky)), 5, kp_color, -1)
                    draw_text_bg(vis, f"{kp_name} {kp_c:.2f}", (int(kx) + 6, int(ky)),
                                 font_scale=0.38, color=kp_color, bg=(30, 30, 30))

    # --- Draw player detections and diagnose each ---
    player_boxes = []
    if player_result is not None and player_result.boxes is not None:
        for box in player_result.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # person class only
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            player_boxes.append([x1, y1, x2, y2])

    for pb in player_boxes:
        x1, y1, x2, y2 = [int(v) for v in pb]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)

        # Diagnose this player box
        if len(pose_boxes) == 0:
            reason = "NO POSE DETS"
            color = (0, 0, 255)
        else:
            ious = _box_iou(list(pb), pose_boxes)
            best_i = int(np.argmax(ious))
            best_iou = float(ious[best_i])

            if best_iou < iou_threshold:
                reason = f"IoU {best_iou:.2f} < {iou_threshold}"
                color = (0, 80, 255)
            elif kp_xy is None:
                reason = f"IoU OK ({best_iou:.2f}) NO KP DATA"
                color = (0, 140, 255)
            else:
                lf = kp_xy[best_i, lf_kp]
                rf = kp_xy[best_i, rf_kp]
                lf_zero = lf[0] == 0.0 and lf[1] == 0.0
                rf_zero = rf[0] == 0.0 and rf[1] == 0.0

                lf_c = float(kp_conf_arr[best_i, lf_kp]) if kp_conf_arr is not None else 1.0
                rf_c = float(kp_conf_arr[best_i, rf_kp]) if kp_conf_arr is not None else 1.0
                lf_ok = not lf_zero and lf_c >= kp_conf_threshold
                rf_ok = not rf_zero and rf_c >= kp_conf_threshold

                if not lf_ok and not rf_ok:
                    reason = f"IoU {best_iou:.2f} LF={lf_c:.2f} RF={rf_c:.2f} BOTH LOW"
                    color = (0, 140, 255)
                else:
                    # Success — draw the ankle midpoint
                    if lf_ok and rf_ok:
                        mx = (lf[0] + rf[0]) / 2.0
                        my = (lf[1] + rf[1]) / 2.0
                    elif lf_ok:
                        mx, my = float(lf[0]), float(lf[1])
                    else:
                        mx, my = float(rf[0]), float(rf[1])
                    cv2.circle(vis, (int(mx), int(my)), 6, (0, 0, 255), -1)
                    cv2.circle(vis, (int(mx), int(my)), 8, (255, 255, 255), 2)
                    reason = f"OK IoU={best_iou:.2f} LF={lf_c:.2f} RF={rf_c:.2f}"
                    color = (0, 220, 0)

        draw_text_bg(vis, reason, (x1, max(y1 - 4, 12)), font_scale=0.4, color=color, bg=(20, 20, 20))

    # --- Legend ---
    legend = [
        ("Green box", (0, 220, 0), "Player detection (YOLO)"),
        ("Blue box", (255, 180, 0), "Pose detection"),
        ("Cyan dot", (255, 255, 0), "Left foot kp"),
        ("Magenta dot", (255, 0, 255), "Right foot kp"),
        ("Red dot", (0, 0, 255), "Ankle midpoint assigned"),
    ]
    lx, ly = 8, h - 10 - len(legend) * 18
    for label, lcolor, desc in legend:
        cv2.circle(vis, (lx + 6, ly + 5), 5, lcolor, -1)
        draw_text_bg(vis, f"{label}: {desc}", (lx + 16, ly + 9), font_scale=0.38, color=(220, 220, 220), bg=(30, 30, 30))
        ly += 18

    return vis


def main():
    args = parse_args()

    print("Loading models...")
    player_model = YOLO(args.model)
    pose_model = YOLO(args.pose_model)

    video_info = VideoInfo.from_path(args.video)
    print(f"Video: {args.video}  {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps")

    frames = list(frame_generator(args.video, start=args.start, end=args.end, stride=args.step))
    frame_indices = list(range(args.start, args.start + len(frames) * args.step, args.step))
    print(f"Loaded {len(frames)} frames. Controls: SPACE/→ next  ←/A prev  Q quit")

    cache: dict[int, np.ndarray] = {}

    def get_vis(fi: int) -> np.ndarray:
        if fi not in cache:
            bgr = frames[fi]
            player_out = player_model.predict(bgr, conf=args.confidence, verbose=False)[0]
            pose_out = pose_model.predict(bgr, conf=0.0001, verbose=False)[0]
            vis = analyze_frame(bgr, player_out, pose_out, args.iou_threshold, args.kp_conf, args.lf_kp, args.rf_kp)
            cv2.putText(vis, f"frame {frame_indices[fi]}  [{fi+1}/{len(frames)}]",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cache[fi] = vis
        return cache[fi]

    win = "Pose ankle debug  [SPACE/→ next  ← prev  Q quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    fi = 0
    while True:
        cv2.imshow(win, get_vis(fi))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord(" "), 83, ord("d")):  # SPACE, →, D
            fi = min(fi + 1, len(frames) - 1)
        elif key in (81, ord("a")):             # ←, A
            fi = max(fi - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
