#!/usr/bin/env python3
"""
diag_result.py  –  result.mp4 원인별 버그 계측 스크립트
airbalance.py 를 수정하지 않고, 원본 로직(NMS 포함)을 인라인으로 재구현하여
프레임별 원인을 집계한다.

출력:
  diag_output/frame_analysis.csv   : 프레임별 상세 로그
  diag_output/captures/f*_*.jpg    : 문제 프레임 캡처 (최대 20개)
  stdout                            : 원인별 발생 횟수 통계
"""
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ── 경로 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "result.mp4"
MODEL_PATH = SCRIPT_DIR / "pose_landmarker.task"
OUT_DIR    = SCRIPT_DIR / "diag_output"

# ── 상수 (원본 airbalance.py 복사) ─────────────────────────────────────────────
LANDMARK_INDEX = {
    "left":  (23, 25, 27),   # left  hip / knee / ankle
    "right": (24, 26, 28),   # right hip / knee / ankle
}
NMS_FRAC      = 0.10   # NMS 발동 거리 임계 (frame_w 비율)
MAX_SNAP_FRAC = 0.10   # Anti-Snap 임계 (frame_h 비율)
MAX_EXTRAP    = 6      # 최대 연속 외삽 프레임
VEL_ALPHA     = 0.45   # 속도 EMA 계수
SAME_POS_FRAC = 0.05   # 두 발목이 이 거리(frame_w 비율) 이하이면 "동일 위치" 판정


# ── 헬퍼 ─────────────────────────────────────────────────────────────────

def _conf(lm) -> float:
    return min(float(getattr(lm, "visibility", 0.0)),
               float(getattr(lm, "presence",   0.0)))

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# ── 원본 extract_leg_readings (airbalance.py 와 동일 로직, 이유별 카운터 추가) ──

def extract_leg_readings_orig(landmarks, h: int, w: int):
    """
    반환:
      readings       : list of dict (left, right 각각)
      skip_seglen    : 세그먼트 길이 부족으로 제거된 side 목록
      skip_flip      : anchor가 ankle 아래(뒤집힘)로 제거된 side 목록
    """
    skip_seglen: List[str] = []
    skip_flip:   List[str] = []
    readings:    List[Dict] = []

    for side in ("left", "right"):
        hip_idx, knee_idx, ankle_idx = LANDMARK_INDEX[side]
        lm_hip   = landmarks[hip_idx]
        lm_knee  = landmarks[knee_idx]
        lm_ankle = landmarks[ankle_idx]

        knee_conf = _conf(lm_knee)
        hip_conf  = _conf(lm_hip)

        hip_f    = (lm_hip.x   * w, lm_hip.y   * h)
        knee_f   = (lm_knee.x  * w, lm_knee.y  * h)
        ankle_f  = (lm_ankle.x * w, lm_ankle.y * h)

        knee_ok = knee_conf >= 0.05
        if knee_ok:
            anchor_f, anchor_conf = knee_f, knee_conf
        else:
            anchor_f, anchor_conf = hip_f, hip_conf

        seg_len = float(np.linalg.norm(np.array(ankle_f) - np.array(anchor_f)))
        if seg_len < h * 0.05:
            skip_seglen.append(side)
            continue
        if anchor_f[1] > ankle_f[1] + h * 0.08:
            skip_flip.append(side)
            continue

        readings.append({
            "side":     side,
            "hip":      (int(hip_f[0]),    int(hip_f[1])),
            "knee":     (int(knee_f[0]),   int(knee_f[1])) if knee_ok else None,
            "ankle":    (int(ankle_f[0]),  int(ankle_f[1])),
            "hip_x":    hip_f[0],
            "hip_y":    hip_f[1],
            "ankle_x":  ankle_f[0],
            "ankle_y":  ankle_f[1],
            "conf":     anchor_conf,
            "raw_left_ankle_x":  lm_ankle.x * w if side == "left"  else None,
            "raw_right_ankle_x": lm_ankle.x * w if side == "right" else None,
        })

    return readings, skip_seglen, skip_flip


# ── 원본 LegMotionFilter (NMS 포함, 상세 이벤트 기록) ────────────────────

def run_filter_orig(readings: List[Dict], frame_h: int, frame_w: int, st: Dict):
    """
    반환:
      result : 최종 display reading 목록 (extrap 포함)
      events : {antisnap_removed, nms_fired, nms_removed_side, extrapolated}
    """
    events = {
        "antisnap_removed": [],
        "nms_fired":        False,
        "nms_removed_side": None,
        "nms_ankle_dist":   0.0,
        "extrapolated":     [],
    }
    snap_thresh = frame_h * MAX_SNAP_FRAC

    # ── Anti-Snap ─────────────────────────────────────────────────────
    valid: List[Dict] = []
    for r in readings:
        side = r["side"]
        prev = st[side]["ankle"]
        if prev is not None:
            d = _dist(r["ankle"], prev)
            if d > snap_thresh:
                events["antisnap_removed"].append(side)
                continue
        else:
            if r["ankle"][1] < frame_h * 0.30:
                events["antisnap_removed"].append(side)
                continue
        valid.append(r)

    # ── 상태 업데이트 ─────────────────────────────────────────────────
    for r in valid:
        side = r["side"]
        prev = st[side]["ankle"]
        if prev is not None:
            raw_vx = float(r["ankle"][0] - prev[0])
            raw_vy = float(r["ankle"][1] - prev[1])
            pvx, pvy = st[side]["vel"]
            st[side]["vel"] = (
                VEL_ALPHA * raw_vx + (1-VEL_ALPHA) * pvx,
                VEL_ALPHA * raw_vy + (1-VEL_ALPHA) * pvy,
            )
        st[side]["ankle"]   = r["ankle"]
        st[side]["hip"]     = r["hip"]
        st[side]["missing"] = 0

    # ── NMS ──────────────────────────────────────────────────────────
    display_valid = list(valid)
    if len(display_valid) == 2:
        r0, r1 = display_valid
        d = _dist(r0["ankle"], r1["ankle"])
        if d < frame_w * NMS_FRAC:
            events["nms_fired"]      = True
            events["nms_ankle_dist"] = d
            if r0["conf"] >= r1["conf"]:
                display_valid = [r0]
                events["nms_removed_side"] = r1["side"]
            else:
                display_valid = [r1]
                events["nms_removed_side"] = r0["side"]

    # ── Extrapolation ─────────────────────────────────────────────────
    display_sides = {r["side"] for r in display_valid}
    result = [{**r, "is_extrap": False} for r in display_valid]

    for side in ("left", "right"):
        if side in display_sides:
            continue
        s = st[side]
        if s["ankle"] is None or s["missing"] >= MAX_EXTRAP:
            s["missing"] += 1
            continue
        vx, vy = s["vel"]
        new_ankle = (
            int(np.clip(s["ankle"][0]+vx, 0, frame_w-1)),
            int(np.clip(s["ankle"][1]+vy, 0, frame_h-1)),
        )
        new_hip = s["hip"] if s["hip"] else new_ankle
        result.append({
            "side":    side,
            "hip":     new_hip,
            "knee":    None,
            "ankle":   new_ankle,
            "hip_x":   float(new_hip[0]),
            "hip_y":   float(new_hip[1]),
            "ankle_x": float(new_ankle[0]),
            "ankle_y": float(new_ankle[1]),
            "conf":    0.0,
            "is_extrap": True,
        })
        events["extrapolated"].append(side)
        s["missing"] += 1

    return result, events


# ── Identity swap 검사 ─────────────────────────────────────────────────────

def check_swap(readings: List[Dict], prev_l, prev_r) -> bool:
    """nearest-neighbor cost 비교. swap이 더 저렴하면 True."""
    lr = next((r for r in readings if r["side"] == "left"),  None)
    rr = next((r for r in readings if r["side"] == "right"), None)
    if not (lr and rr) or prev_l is None or prev_r is None:
        return False
    cost_keep = _dist(lr["ankle"], prev_l) + _dist(rr["ankle"], prev_r)
    cost_swap = _dist(lr["ankle"], prev_r) + _dist(rr["ankle"], prev_l)
    return cost_swap < cost_keep


# ── 문제 프레임 시각화 ──────────────────────────────────────────────────────

def draw_debug(frame: np.ndarray, raw: List[Dict], filt: List[Dict],
               events: Dict, skip_seg: List[str], skip_flip: List[str],
               id_swap: bool, ankles_same: bool,
               frame_idx: int, h: int, w: int) -> np.ndarray:
    vis = frame.copy()

    # Raw ankle : 초록(left) / 빨강(right)
    for r in raw:
        col = (0, 220, 80) if r["side"] == "left" else (80, 80, 220)
        cv2.circle(vis, r["ankle"], 14, col, 3)
        cv2.putText(vis, f"RAW {r['side'][0].upper()}",
                    (r["ankle"][0]+16, r["ankle"][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # Filter result : 노랑(extrap) / 청록(left-ok) / 자홍(right-ok)
    for r in filt:
        if r["is_extrap"]:
            col = (0, 200, 255)
            tag = "EXTRAP"
        elif r["side"] == "left":
            col = (255, 200, 0)
            tag = "L-OK"
        else:
            col = (255, 0, 200)
            tag = "R-OK"
        cv2.circle(vis, r["ankle"], 20, col, 3)
        cv2.putText(vis, tag,
                    (r["ankle"][0]+20, r["ankle"][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # 원인 레이블 (상단 정보 패널 위에 덮어쓰기)
    causes: List[str] = []
    if skip_seg:                   causes.append(f"SKIP_SEGLEN:{skip_seg}")
    if skip_flip:                  causes.append(f"SKIP_FLIP:{skip_flip}")
    if events["antisnap_removed"]: causes.append(f"ANTISNAP:{events['antisnap_removed']}")
    if events["nms_fired"]:
        causes.append(
            f"NMS:rm={events['nms_removed_side']} dist={events['nms_ankle_dist']:.0f}px")
    if events["extrapolated"]:     causes.append(f"EXTRAP:{events['extrapolated']}")
    if ankles_same:                causes.append("SAME_POS")
    if id_swap:                    causes.append("ID_SWAP")

    banner_h = max(50, 28 * len(causes) + 12)
    cv2.rectangle(vis, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.putText(vis, f"f={frame_idx}  raw={len(raw)}  filt={len(filt)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    for i, c in enumerate(causes):
        cv2.putText(vis, c, (8, 46 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

    return vis


# ── 메인 ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)
    caps_dir = OUT_DIR / "captures"
    caps_dir.mkdir(exist_ok=True)

    # MediaPipe 초기화
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.30,
        output_segmentation_masks=False,
    )
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"분석 대상: {VIDEO_PATH.name}")
    print(f"  해상도: {frame_w}x{frame_h}  fps={fps:.2f}  총 {total}프레임 ({total/fps:.1f}s)")

    SAME_POS_THRESH = frame_w * SAME_POS_FRAC

    # 원본 필터 상태
    st = {
        side: {"ankle": None, "hip": None, "vel": (0.0, 0.0), "missing": 0}
        for side in ("left", "right")
    }

    # 누적 카운터
    cnt: Dict[str, int] = {
        "extract_skip_seglen": 0,
        "extract_skip_flip":   0,
        "antisnap_removed":    0,
        "nms_fired":           0,
        "extrapolation_used":  0,
        "ankles_same_pos":     0,
        "identity_swap":       0,
        "frames_0vector":      0,
        "frames_1vector":      0,
        "frames_2vector":      0,
        "frames_no_pose":      0,
    }

    prev_left_ankle  = None
    prev_right_ankle = None
    csv_rows:       List[Dict] = []
    problem_frames: List[Tuple] = []   # (frame_idx, reason_str, img)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(round(frame_index * 1000.0 / fps))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_res  = detector.detect_for_video(mp_img, ts_ms)
        landmarks = mp_res.pose_landmarks[0] if mp_res.pose_landmarks else None

        # ── extract_leg_readings ──────────────────────────────────────
        raw_readings: List[Dict] = []
        skip_seglen_sides: List[str] = []
        skip_flip_sides:   List[str] = []

        if landmarks is not None:
            raw_readings, skip_seglen_sides, skip_flip_sides = \
                extract_leg_readings_orig(landmarks, frame_h, frame_w)
        else:
            cnt["frames_no_pose"] += 1

        # ── LegMotionFilter (원본, NMS 포함) ─────────────────────────
        filter_result, events = run_filter_orig(
            raw_readings, frame_h, frame_w, st)

        # ── 보조 지표 ─────────────────────────────────────────────────
        left_raw  = next((r for r in raw_readings if r["side"] == "left"),  None)
        right_raw = next((r for r in raw_readings if r["side"] == "right"), None)

        ankles_same = (
            bool(left_raw and right_raw)
            and _dist(left_raw["ankle"], right_raw["ankle"]) < SAME_POS_THRESH
        )
        id_swap = check_swap(raw_readings, prev_left_ankle, prev_right_ankle)

        # ── 통계 누적 ─────────────────────────────────────────────────
        cnt["extract_skip_seglen"] += len(skip_seglen_sides)
        cnt["extract_skip_flip"]   += len(skip_flip_sides)
        cnt["antisnap_removed"]    += len(events["antisnap_removed"])
        if events["nms_fired"]:      cnt["nms_fired"]           += 1
        cnt["extrapolation_used"]  += len(events["extrapolated"])
        if ankles_same:              cnt["ankles_same_pos"]      += 1
        if id_swap:                  cnt["identity_swap"]        += 1

        n_vec = len(filter_result)
        if   n_vec == 0: cnt["frames_0vector"] += 1
        elif n_vec == 1: cnt["frames_1vector"] += 1
        else:            cnt["frames_2vector"] += 1

        # ── prev 업데이트 ─────────────────────────────────────────────
        for r in filter_result:
            if not r.get("is_extrap", False):
                if r["side"] == "left":  prev_left_ankle  = r["ankle"]
                if r["side"] == "right": prev_right_ankle = r["ankle"]

        # ── CSV 행 구성 ───────────────────────────────────────────────
        csv_rows.append({
            "frame_idx":        frame_index,
            "ts_ms":            ts_ms,
            "pose_detected":    1 if landmarks else 0,
            # raw – left
            "left_hip_x":       int(left_raw["hip_x"])    if left_raw  else -1,
            "left_hip_y":       int(left_raw["hip_y"])    if left_raw  else -1,
            "left_ankle_x":     int(left_raw["ankle_x"])  if left_raw  else -1,
            "left_ankle_y":     int(left_raw["ankle_y"])  if left_raw  else -1,
            "left_conf":        round(left_raw["conf"],4) if left_raw  else -1,
            # raw – right
            "right_hip_x":      int(right_raw["hip_x"])   if right_raw else -1,
            "right_hip_y":      int(right_raw["hip_y"])   if right_raw else -1,
            "right_ankle_x":    int(right_raw["ankle_x"]) if right_raw else -1,
            "right_ankle_y":    int(right_raw["ankle_y"]) if right_raw else -1,
            "right_conf":       round(right_raw["conf"],4)if right_raw else -1,
            # counts
            "raw_readings_n":   len(raw_readings),
            "filter_readings_n":len(filter_result),
            "extrap_sides":     ",".join(events["extrapolated"]),
            # causes
            "skip_seglen":      ",".join(skip_seglen_sides),
            "skip_flip":        ",".join(skip_flip_sides),
            "antisnap_sides":   ",".join(events["antisnap_removed"]),
            "nms_fired":        int(events["nms_fired"]),
            "nms_removed_side": events["nms_removed_side"] or "",
            "nms_ankle_dist_px":round(events["nms_ankle_dist"],1),
            "ankles_same_pos":  int(ankles_same),
            "identity_swap":    int(id_swap),
        })

        # ── 문제 프레임 캡처 (최대 20개) ─────────────────────────────
        is_problem = (
            len(raw_readings) < 2
            or events["nms_fired"]
            or events["antisnap_removed"]
            or events["extrapolated"]
            or ankles_same
            or id_swap
        )
        if is_problem and len(problem_frames) < 20:
            reasons: List[str] = []
            if len(raw_readings) < 2:         reasons.append(f"raw{len(raw_readings)}")
            if skip_seglen_sides:              reasons.append("SEGLEN")
            if skip_flip_sides:               reasons.append("FLIP")
            if events["antisnap_removed"]:    reasons.append("ANTISNAP")
            if events["nms_fired"]:           reasons.append("NMS")
            if events["extrapolated"]:        reasons.append("EXTRAP")
            if ankles_same:                   reasons.append("SAMEPOS")
            if id_swap:                       reasons.append("SWAP")
            vis = draw_debug(
                frame, raw_readings, filter_result, events,
                skip_seglen_sides, skip_flip_sides,
                id_swap, ankles_same,
                frame_index, frame_h, frame_w,
            )
            problem_frames.append((frame_index, "_".join(reasons), vis))

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  처리 중: {frame_index}/{total}", flush=True)

    cap.release()
    detector.close()

    total_frames = frame_index

    # ── CSV 저장 ───────────────────────────────────────────────────────
    csv_path = OUT_DIR / "frame_analysis.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"\nCSV 저장: {csv_path}  ({len(csv_rows)}행)")

    # ── 캡처 이미지 저장 ───────────────────────────────────────────────
    for fidx, reason, img in problem_frames:
        fname = caps_dir / f"f{fidx:05d}_{reason[:60]}.jpg"
        cv2.imwrite(str(fname), img)
    print(f"캡처 저장: {caps_dir}  ({len(problem_frames)}개)")

    # ── 최종 통계 출력 ─────────────────────────────────────────────────
    def pct(n): return f"{100*n/total_frames:.1f}%"

    print(f"\n{'='*58}")
    print(f"  분석 완료: {total_frames}프레임")
    print(f"{'='*58}")
    print(f"  [벡터 개수 분포]")
    print(f"  2개 (정상):              {cnt['frames_2vector']:5d}  {pct(cnt['frames_2vector'])}")
    print(f"  1개 (버그):              {cnt['frames_1vector']:5d}  {pct(cnt['frames_1vector'])}")
    print(f"  0개 (완전 누락):         {cnt['frames_0vector']:5d}  {pct(cnt['frames_0vector'])}")
    print(f"  포즈 미감지:             {cnt['frames_no_pose']:5d}  {pct(cnt['frames_no_pose'])}")
    print(f"{'─'*58}")
    print(f"  [원인별 발생 횟수 (프레임 수 아닌 이벤트 수)]")
    print(f"  extract: seg_len 미달 제거:    {cnt['extract_skip_seglen']:5d}")
    print(f"  extract: anchor>ankle 뒤집힘:  {cnt['extract_skip_flip']:5d}")
    print(f"  Anti-Snap 제거 (이벤트):       {cnt['antisnap_removed']:5d}")
    print(f"  NMS 발동 (프레임):             {cnt['nms_fired']:5d}  {pct(cnt['nms_fired'])}")
    print(f"  Extrapolation 사용 (이벤트):   {cnt['extrapolation_used']:5d}")
    print(f"  두 발목 동일 위치 (<{int(SAME_POS_FRAC*100)}%w):  {cnt['ankles_same_pos']:5d}  {pct(cnt['ankles_same_pos'])}")
    print(f"  Identity swap 감지 (프레임):   {cnt['identity_swap']:5d}  {pct(cnt['identity_swap'])}")
    print(f"{'='*58}")
    print(f"\n  ※ NMS가 {pct(cnt['nms_fired'])} 프레임에서 발동 → 1-vector 현상의 직접 원인")
    print(f"  ※ Identity swap {pct(cnt['identity_swap'])} 프레임 → 같은 다리 몰림 원인 후보")
    print(f"\n  문제 프레임 캡처: {caps_dir}")
    print(f"  상세 로그 CSV:   {csv_path}")


if __name__ == "__main__":
    main()
