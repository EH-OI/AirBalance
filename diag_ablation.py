#!/usr/bin/env python3
"""
diag_ablation.py  –  Anti-Snap / NMS 제거 실험 A·B·C
+ MediaPipe 원본 랜드마크 품질 검증

실험:
  A  : Anti-Snap 완전 비활성화 (MAX_SNAP_FRAC=999)
  B  : NMS 완전 비활성화
  C  : Anti-Snap + NMS 둘 다 비활성화
  원본: 기존 로직 (baseline)

추가 검증:
  - MediaPipe 원본 left/right ankle 좌표가 same-pos인 프레임 수
  - 조건: dist(lm[27], lm[28]) < 5% frame_w

출력:
  diag_output/ablation_summary.txt     : 비교 통계
  diag_output/ablation_rawankle.csv    : same-pos 프레임별 원본 좌표
"""
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "result.mp4"
MODEL_PATH = SCRIPT_DIR / "pose_landmarker.task"
OUT_DIR    = SCRIPT_DIR / "diag_output"

LANDMARK_INDEX = {"left": (23, 25, 27), "right": (24, 26, 28)}
MAX_EXTRAP          = 6
VEL_ALPHA           = 0.45
SAME_POS_FRAC       = 0.05   # frame_w 비율
CONF_FREEZE_THRESH  = 0.50   # 실험 D: ankle conf 이하면 freeze
MIN_ANKLE_SEP_FRAC  = 0.05   # 실험 D: 양 발목 거리 이하면 프레임 전체 freeze


def _conf(lm) -> float:
    return min(float(getattr(lm, "visibility", 0.0)),
               float(getattr(lm, "presence",   0.0)))

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


# ── extract_leg_readings (모든 실험 공통) ─────────────────────────────────

def extract_readings(landmarks, h: int, w: int) -> List[Dict]:
    readings: List[Dict] = []
    for side in ("left", "right"):
        hip_idx, knee_idx, ankle_idx = LANDMARK_INDEX[side]
        lm_hip   = landmarks[hip_idx]
        lm_knee  = landmarks[knee_idx]
        lm_ankle = landmarks[ankle_idx]

        knee_conf = _conf(lm_knee)
        hip_conf  = _conf(lm_hip)
        hip_f     = (lm_hip.x * w,   lm_hip.y * h)
        knee_f    = (lm_knee.x * w,  lm_knee.y * h)
        ankle_f   = (lm_ankle.x * w, lm_ankle.y * h)

        knee_ok = knee_conf >= 0.05
        anchor_f, anchor_conf = (knee_f, knee_conf) if knee_ok else (hip_f, hip_conf)

        seg_len = float(np.linalg.norm(np.array(ankle_f) - np.array(anchor_f)))
        if seg_len < h * 0.05:
            continue
        if anchor_f[1] > ankle_f[1] + h * 0.08:
            continue

        readings.append({
            "side":   side,
            "hip":    (int(hip_f[0]),    int(hip_f[1])),
            "ankle":  (int(ankle_f[0]),  int(ankle_f[1])),
            "conf":   anchor_conf,
        })
    return readings


# ── 실험별 필터 (Anti-Snap, NMS 스위치 포함) ─────────────────────────────

def run_filter(
    readings: List[Dict],
    frame_h: int,
    frame_w: int,
    st: Dict,
    disable_antisnap: bool = False,
    disable_nms: bool = False,
) -> Tuple[List[Dict], Dict]:
    events = {
        "antisnap_removed": [],
        "nms_fired": False,
        "nms_removed_side": None,
        "extrapolated": [],
    }
    snap_thresh = frame_h * (999 if disable_antisnap else 0.10)

    # Anti-Snap
    valid: List[Dict] = []
    for r in readings:
        side = r["side"]
        prev = st[side]["ankle"]
        if prev is not None:
            if _dist(r["ankle"], prev) > snap_thresh:
                events["antisnap_removed"].append(side)
                continue
        else:
            if not disable_antisnap and r["ankle"][1] < frame_h * 0.30:
                events["antisnap_removed"].append(side)
                continue
        valid.append(r)

    # State update
    for r in valid:
        side = r["side"]
        prev = st[side]["ankle"]
        if prev is not None:
            raw_vx = float(r["ankle"][0] - prev[0])
            raw_vy = float(r["ankle"][1] - prev[1])
            pvx, pvy = st[side]["vel"]
            st[side]["vel"] = (
                VEL_ALPHA * raw_vx + (1 - VEL_ALPHA) * pvx,
                VEL_ALPHA * raw_vy + (1 - VEL_ALPHA) * pvy,
            )
        st[side]["ankle"]   = r["ankle"]
        st[side]["hip"]     = r["hip"]
        st[side]["missing"] = 0

    # NMS
    display_valid = list(valid)
    if not disable_nms and len(display_valid) == 2:
        r0, r1 = display_valid
        if _dist(r0["ankle"], r1["ankle"]) < frame_w * 0.10:
            events["nms_fired"] = True
            if r0["conf"] >= r1["conf"]:
                display_valid = [r0]
                events["nms_removed_side"] = r1["side"]
            else:
                display_valid = [r1]
                events["nms_removed_side"] = r0["side"]

    # Extrapolation
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
            int(np.clip(s["ankle"][0] + vx, 0, frame_w - 1)),
            int(np.clip(s["ankle"][1] + vy, 0, frame_h - 1)),
        )
        new_hip = s["hip"] if s["hip"] else new_ankle
        result.append({
            "side": side, "hip": new_hip, "ankle": new_ankle,
            "conf": 0.0, "is_extrap": True,
        })
        events["extrapolated"].append(side)
        s["missing"] += 1

    return result, events


# ── 실험 D: Identity assignment ──────────────────────────────────────────

def assign_identities_d(candidates: List[Dict], st: Dict) -> List[Dict]:
    """Nearest-neighbor assignment (airbalance.py 와 동일 로직)."""
    if len(candidates) != 2:
        return candidates
    left_c  = next((r for r in candidates if r["side"] == "left"),  None)
    right_c = next((r for r in candidates if r["side"] == "right"), None)
    if not (left_c and right_c):
        return candidates
    prev_l = st["left"]["ankle"]
    prev_r = st["right"]["ankle"]
    if prev_l is None or prev_r is None:
        return candidates
    cost_keep = _dist(left_c["ankle"], prev_l) + _dist(right_c["ankle"], prev_r)
    cost_swap = _dist(left_c["ankle"], prev_r) + _dist(right_c["ankle"], prev_l)
    if cost_swap < cost_keep:
        return [{**right_c, "side": "left"}, {**left_c, "side": "right"}]
    return candidates


# ── 실험 D: Freeze 필터 (Anti-Snap·NMS 제거, Identity+Freeze 추가) ─────────

def run_filter_freeze(
    readings: List[Dict],
    frame_h: int,
    frame_w: int,
    st: Dict,
) -> Tuple[List[Dict], Dict]:
    events = {
        "antisnap_removed": [],
        "nms_fired": False,
        "nms_removed_side": None,
        "extrapolated": [],
        "frozen": [],
    }

    # 1. Identity assignment
    candidates = assign_identities_d(readings, st)

    # 2. 프레임 품질 판정
    left_c  = next((r for r in candidates if r["side"] == "left"),  None)
    right_c = next((r for r in candidates if r["side"] == "right"), None)
    frame_bad = (
        left_c is not None and right_c is not None
        and _dist(left_c["ankle"], right_c["ankle"]) < frame_w * MIN_ANKLE_SEP_FRAC
    )

    # 3. Freeze or Accept
    result: List[Dict] = []
    active_sides: set  = set()

    for r in candidates:
        quality_bad = frame_bad or r["conf"] < CONF_FREEZE_THRESH
        s = st[r["side"]]

        if quality_bad:
            if s["ankle"] is not None:
                result.append({
                    **r,
                    "ankle":     s["ankle"],
                    "hip":       s["hip"] if s["hip"] else s["ankle"],
                    "conf":      0.0,
                    "is_extrap": False,
                    "is_frozen": True,
                })
                s["missing"] = 0
                active_sides.add(r["side"])
                events["frozen"].append(r["side"])
            else:
                s["missing"] += 1
        else:
            if s["ankle"] is not None:
                vx = float(r["ankle"][0] - s["ankle"][0])
                vy = float(r["ankle"][1] - s["ankle"][1])
                pvx, pvy = s["vel"]
                s["vel"] = (
                    VEL_ALPHA * vx + (1 - VEL_ALPHA) * pvx,
                    VEL_ALPHA * vy + (1 - VEL_ALPHA) * pvy,
                )
            s["ankle"]   = r["ankle"]
            s["hip"]     = r["hip"]
            s["missing"] = 0
            result.append({**r, "is_extrap": False, "is_frozen": False})
            active_sides.add(r["side"])

    # 4. Extrapolation
    for side in ("left", "right"):
        if side in active_sides:
            continue
        s = st[side]
        if s["ankle"] is None or s["missing"] >= MAX_EXTRAP:
            s["missing"] += 1
            continue
        vx, vy = s["vel"]
        new_ankle = (
            int(np.clip(s["ankle"][0] + vx, 0, frame_w - 1)),
            int(np.clip(s["ankle"][1] + vy, 0, frame_h - 1)),
        )
        new_hip = s["hip"] if s["hip"] else new_ankle
        result.append({
            "side": side, "hip": new_hip, "ankle": new_ankle,
            "conf": 0.0, "is_extrap": True, "is_frozen": False,
        })
        events["extrapolated"].append(side)
        s["missing"] += 1

    return result, events


# ── 실험 E·F: conf 기반 freeze 필터 ──────────────────────────────────────
# E : 원본 Anti-Snap·NMS 유지 + ankle_conf < 0.5 → prev ankle 사용
# F : Anti-Snap·NMS 제거     + ankle_conf < 0.5 → prev ankle 사용 (sep 조건 없음)

def run_filter_conf_freeze(
    readings: List[Dict],
    frame_h: int,
    frame_w: int,
    st: Dict,
    disable_antisnap: bool = False,
    disable_nms: bool = False,
    conf_thresh: float = 0.50,
) -> Tuple[List[Dict], Dict]:
    """
    ankle_conf < conf_thresh 인 reading 을 현재 MediaPipe 좌표 대신
    직전 정상 발목 좌표로 교체한다.
    Anti-Snap / NMS 는 스위치로 켜고 끌 수 있다.
    """
    events = {
        "antisnap_removed": [],
        "nms_fired": False,
        "nms_removed_side": None,
        "extrapolated": [],
        "frozen": [],
    }
    snap_thresh = frame_h * (999 if disable_antisnap else 0.10)

    # ── Anti-Snap ──────────────────────────────────────────────────────
    after_snap: List[Dict] = []
    for r in readings:
        side = r["side"]
        prev = st[side]["ankle"]
        if prev is not None:
            if _dist(r["ankle"], prev) > snap_thresh:
                events["antisnap_removed"].append(side)
                continue
        else:
            if not disable_antisnap and r["ankle"][1] < frame_h * 0.30:
                events["antisnap_removed"].append(side)
                continue
        after_snap.append(r)

    # ── conf freeze: conf < thresh → prev ankle 사용 ──────────────────
    result_pre_nms: List[Dict] = []
    for r in after_snap:
        side = r["side"]
        s    = st[side]
        if r["conf"] < conf_thresh and s["ankle"] is not None:
            # 직전 정상 좌표로 교체
            result_pre_nms.append({
                **r,
                "ankle":     s["ankle"],
                "hip":       s["hip"] if s["hip"] else s["ankle"],
                "conf":      0.0,
                "is_frozen": True,
            })
            events["frozen"].append(side)
        else:
            result_pre_nms.append({**r, "is_frozen": False})

    # ── State update (conf-freeze 된 좌표는 state 를 갱신하지 않음) ───
    for r in result_pre_nms:
        side = r["side"]
        s    = st[side]
        if r.get("is_frozen"):
            s["missing"] = 0          # 다리는 존재함, 좌표만 고정
            continue
        prev = s["ankle"]
        if prev is not None:
            vx = float(r["ankle"][0] - prev[0])
            vy = float(r["ankle"][1] - prev[1])
            pvx, pvy = s["vel"]
            s["vel"] = (
                VEL_ALPHA * vx + (1 - VEL_ALPHA) * pvx,
                VEL_ALPHA * vy + (1 - VEL_ALPHA) * pvy,
            )
        s["ankle"]   = r["ankle"]
        s["hip"]     = r["hip"]
        s["missing"] = 0

    # ── NMS ───────────────────────────────────────────────────────────
    display_valid = list(result_pre_nms)
    if not disable_nms and len(display_valid) == 2:
        r0, r1 = display_valid
        if _dist(r0["ankle"], r1["ankle"]) < frame_w * 0.10:
            events["nms_fired"] = True
            if r0["conf"] >= r1["conf"]:
                display_valid = [r0]
                events["nms_removed_side"] = r1["side"]
            else:
                display_valid = [r1]
                events["nms_removed_side"] = r0["side"]

    display_sides = {r["side"] for r in display_valid}
    result = [{**r, "is_extrap": False} for r in display_valid]

    # ── Extrapolation ─────────────────────────────────────────────────
    for side in ("left", "right"):
        if side in display_sides:
            continue
        s = st[side]
        if s["ankle"] is None or s["missing"] >= MAX_EXTRAP:
            s["missing"] += 1
            continue
        vx, vy = s["vel"]
        new_ankle = (
            int(np.clip(s["ankle"][0] + vx, 0, frame_w - 1)),
            int(np.clip(s["ankle"][1] + vy, 0, frame_h - 1)),
        )
        new_hip = s["hip"] if s["hip"] else new_ankle
        result.append({
            "side": side, "hip": new_hip, "ankle": new_ankle,
            "conf": 0.0, "is_extrap": True, "is_frozen": False,
        })
        events["extrapolated"].append(side)
        s["missing"] += 1

    return result, events


# ── Identity swap 검사 (원본 reading 기준) ──────────────────────────────

def check_swap(readings: List[Dict], prev_l, prev_r) -> bool:
    lr = next((r for r in readings if r["side"] == "left"),  None)
    rr = next((r for r in readings if r["side"] == "right"), None)
    if not (lr and rr) or prev_l is None or prev_r is None:
        return False
    cost_keep = _dist(lr["ankle"], prev_l) + _dist(rr["ankle"], prev_r)
    cost_swap = _dist(lr["ankle"], prev_r) + _dist(rr["ankle"], prev_l)
    return cost_swap < cost_keep


# ── 통계 카운터 초기화 ────────────────────────────────────────────────────

def make_counter() -> Dict:
    return {"n0": 0, "n1": 0, "n2": 0,
            "antisnap": 0, "nms": 0, "extrap": 0,
            "frozen": 0, "swap": 0, "samepos": 0}


def tally(counter: Dict, result: List[Dict], events: Dict,
          swap: bool, samepos: bool) -> None:
    n = len(result)
    if   n == 0: counter["n0"] += 1
    elif n == 1: counter["n1"] += 1
    else:        counter["n2"] += 1
    counter["antisnap"] += len(events.get("antisnap_removed", []))
    if events.get("nms_fired"): counter["nms"] += 1
    counter["extrap"] += len(events.get("extrapolated", []))
    counter["frozen"] += len(events.get("frozen", []))
    if swap:    counter["swap"]    += 1
    if samepos: counter["samepos"] += 1


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

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

    cap     = cv2.VideoCapture(str(VIDEO_PATH))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"대상: {VIDEO_PATH.name}  {frame_w}x{frame_h}  {total}f  {fps:.1f}fps")

    SAME_POS_THRESH = frame_w * SAME_POS_FRAC

    def make_st():
        return {s: {"ankle": None, "hip": None, "vel": (0.0, 0.0), "missing": 0}
                for s in ("left", "right")}

    # 실험별 독립 state
    # cfg = (disable_antisnap, disable_nms)  / None = 별도 함수 사용
    experiments = {
        "원본":            {"st": make_st(), "cfg": (False, False), "mode": "orig",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
        "A(no_antisnap)": {"st": make_st(), "cfg": (True,  False), "mode": "orig",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
        "B(no_nms)":      {"st": make_st(), "cfg": (False, True),  "mode": "orig",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
        "C(no_both)":     {"st": make_st(), "cfg": (True,  True),  "mode": "orig",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
        "D(freeze)":      {"st": make_st(), "cfg": None,            "mode": "freeze_d", "cnt": make_counter(), "prev_l": None, "prev_r": None},
        # E: 원본(Anti-Snap+NMS 유지) + conf<0.5 freeze
        "E(orig+conf)":   {"st": make_st(), "cfg": (False, False), "mode": "conf",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
        # F: Anti-Snap·NMS 제거 + conf<0.5 freeze (sep 조건 없음)
        "F(no_AS+conf)":  {"st": make_st(), "cfg": (True,  True),  "mode": "conf",  "cnt": make_counter(), "prev_l": None, "prev_r": None},
    }

    # MediaPipe 원본 랜드마크 품질 검사
    rawankle_rows: List[Dict] = []
    raw_samepos_cnt = 0  # MediaPipe 원본부터 동일 위치인 경우
    total_pose = 0
    no_pose    = 0

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(round(frame_index * 1000.0 / fps))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_res  = detector.detect_for_video(mp_img, ts_ms)
        lms     = mp_res.pose_landmarks[0] if mp_res.pose_landmarks else None

        if lms is None:
            no_pose += 1
            # 필터 상태 유지하되 아무것도 하지 않음
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"  {frame_index}/{total}", flush=True)
            continue

        total_pose += 1

        # ── MediaPipe 원본 발목 좌표 (필터링 전) ─────────────────────
        lm_la = lms[27]  # left  ankle
        lm_ra = lms[28]  # right ankle
        la_raw = (lm_la.x * frame_w, lm_la.y * frame_h)
        ra_raw = (lm_ra.x * frame_w, lm_ra.y * frame_h)
        raw_dist = _dist(la_raw, ra_raw)
        raw_same  = raw_dist < SAME_POS_THRESH

        lm_lh = lms[23]  # left  hip
        lm_rh = lms[24]  # right hip
        la_conf = _conf(lm_la)
        ra_conf = _conf(lm_ra)

        if raw_same:
            raw_samepos_cnt += 1
            rawankle_rows.append({
                "frame_idx":    frame_index,
                "ts_ms":        ts_ms,
                "la_x":         round(la_raw[0], 1),
                "la_y":         round(la_raw[1], 1),
                "ra_x":         round(ra_raw[0], 1),
                "ra_y":         round(ra_raw[1], 1),
                "dist_px":      round(raw_dist, 1),
                "la_conf":      round(la_conf, 4),
                "ra_conf":      round(ra_conf, 4),
                "lh_x":         round(lm_lh.x * frame_w, 1),
                "rh_x":         round(lm_rh.x * frame_w, 1),
            })

        # ── extract_leg_readings (공통) ──────────────────────────────
        raw_readings = extract_readings(lms, frame_h, frame_w)

        # samepos 는 raw_readings 기준 (공통)
        lr_raw = next((r for r in raw_readings if r["side"] == "left"),  None)
        rr_raw = next((r for r in raw_readings if r["side"] == "right"), None)
        samepos_raw = bool(lr_raw and rr_raw) and _dist(lr_raw["ankle"], rr_raw["ankle"]) < SAME_POS_THRESH

        # ── 각 실험 실행 ─────────────────────────────────────────────
        for name, exp in experiments.items():
            mode = exp["mode"]
            if mode == "freeze_d":
                result, events = run_filter_freeze(
                    raw_readings, frame_h, frame_w, exp["st"])
            elif mode == "conf":
                dis_as, dis_nms = exp["cfg"]
                result, events = run_filter_conf_freeze(
                    raw_readings, frame_h, frame_w, exp["st"],
                    disable_antisnap=dis_as,
                    disable_nms=dis_nms,
                    conf_thresh=CONF_FREEZE_THRESH,
                )
            else:  # orig
                dis_as, dis_nms = exp["cfg"]
                result, events = run_filter(
                    raw_readings, frame_h, frame_w, exp["st"],
                    disable_antisnap=dis_as,
                    disable_nms=dis_nms,
                )

            # swap 검사는 raw_readings 기준 (필터 독립적 지표)
            swap = check_swap(raw_readings, exp["prev_l"], exp["prev_r"])

            tally(exp["cnt"], result, events, swap, samepos_raw)

            for r in result:
                if not r.get("is_extrap", False):
                    if r["side"] == "left":  exp["prev_l"] = r["ankle"]
                    if r["side"] == "right": exp["prev_r"] = r["ankle"]

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  {frame_index}/{total}", flush=True)

    cap.release()
    detector.close()

    T = frame_index   # 전체 프레임

    def pct(n, denom=None):
        d = denom if denom is not None else T
        return f"{100*n/d:.1f}%" if d else "—"

    # ── 결과 출력 ─────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append(f"{'='*68}")
    lines.append(f"  ablation 실험 결과  |  {VIDEO_PATH.name}  {T}프레임")
    lines.append(f"{'='*68}")
    lines.append(f"  MediaPipe 포즈 감지: {total_pose}/{T}  미감지: {no_pose}")
    lines.append(f"{'─'*68}")

    header = f"  {'지표':<26}" + "".join(f"{'exp':>10}" for exp in experiments)
    lines.append(f"  {'지표':<26}" +
                 "".join(f"{n:>14}" for n in experiments))
    lines.append(f"{'─'*68}")

    rows_def = [
        ("2-vector 프레임",       "n2"),
        ("1-vector 프레임",       "n1"),
        ("0-vector 프레임",       "n0"),
        ("Anti-Snap 이벤트",      "antisnap"),
        ("NMS 발동 프레임",       "nms"),
        ("Freeze 이벤트(D전용)",  "frozen"),
        ("Extrap 이벤트",         "extrap"),
        ("Swap 프레임",           "swap"),
        ("SamePos 프레임",        "samepos"),
    ]
    for label, key in rows_def:
        vals = "".join(
            f"{exp['cnt'][key]:6d}({pct(exp['cnt'][key]):>6})"
            for exp in experiments.values()
        )
        lines.append(f"  {label:<26}{vals}")

    lines.append(f"{'─'*68}")
    lines.append(f"  [MediaPipe 원본 랜드마크 품질]")
    lines.append(f"  포즈 감지 {total_pose}프레임 중:")
    lines.append(f"    left ankle == right ankle (<5%w={SAME_POS_THRESH:.0f}px):")
    lines.append(f"      {raw_samepos_cnt}프레임  ({pct(raw_samepos_cnt, total_pose)})")
    lines.append(f"")
    lines.append(f"  ※ 이 {raw_samepos_cnt}프레임은 MediaPipe 원본부터 손상됨.")
    lines.append(f"     후처리(Anti-Snap/NMS/Identity)로 복구 불가.")
    lines.append(f"{'='*68}")

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)

    # ── 파일 저장 ─────────────────────────────────────────────────────
    txt_path = OUT_DIR / "ablation_summary.txt"
    txt_path.write_text(summary_txt + "\n", encoding="utf-8")
    print(f"\n통계 저장: {txt_path}")

    if rawankle_rows:
        csv_path = OUT_DIR / "ablation_rawankle.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rawankle_rows[0].keys()))
            w.writeheader()
            w.writerows(rawankle_rows)
        print(f"raw ankle CSV: {csv_path}  ({len(rawankle_rows)}행)")

        # 상위 10개 좌표 출력
        print(f"\n  [원본 랜드마크 손상 샘플 (dist 큰 순)]")
        for r in sorted(rawankle_rows, key=lambda x: -x["dist_px"])[:10]:
            same = "EXACT" if r["dist_px"] < 2 else "NEAR"
            print(f"    f={r['frame_idx']:05d} "
                  f"LA=({r['la_x']:.0f},{r['la_y']:.0f}) "
                  f"RA=({r['ra_x']:.0f},{r['ra_y']:.0f}) "
                  f"dist={r['dist_px']:.1f}px "
                  f"la_conf={r['la_conf']:.3f} ra_conf={r['ra_conf']:.3f}  [{same}]")


if __name__ == "__main__":
    main()
