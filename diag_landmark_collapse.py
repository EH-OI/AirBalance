#!/usr/bin/env python3
"""
diag_landmark_collapse.py
SamePos 프레임(left ankle ≈ right ankle)에서
hip / knee / ankle 각 관절 쌍의 left-right 거리를 측정하여
"발목만 붕괴" vs "무릎+발목 같이 붕괴" vs "전체 붕괴"를 분류한다.

출력:
  diag_output/collapse_frames.csv    : SamePos 프레임별 관절 거리 전체
  diag_output/collapse_summary.txt   : 붕괴 패턴 통계
  diag_output/collapse_caps/         : 캡처 이미지 (최대 15개)
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

# left/right 쌍 인덱스
JOINTS = {
    "hip":   (23, 24),
    "knee":  (25, 26),
    "ankle": (27, 28),
}
SAME_POS_FRAC = 0.05   # frame_w 비율 → threshold


def _conf(lm) -> float:
    return min(float(getattr(lm, "visibility", 0.0)),
               float(getattr(lm, "presence",   0.0)))

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def draw_collapse(frame, lms, frame_w, frame_h, thresh, dists, pattern, fidx):
    """붕괴 패턴을 시각화한다."""
    vis = frame.copy()

    joint_colors = {"hip": (0,200,100), "knee": (0,150,255), "ankle": (0,50,255)}
    for name, (li, ri) in JOINTS.items():
        lm_l = lms[li]; lm_r = lms[ri]
        pl = (int(lm_l.x*frame_w), int(lm_l.y*frame_h))
        pr = (int(lm_r.x*frame_w), int(lm_r.y*frame_h))
        col = joint_colors[name]
        collapsed = dists[name] < thresh
        thick = 4 if collapsed else 2
        ring  = 16 if collapsed else 8
        cv2.circle(vis, pl, ring, col, thick)
        cv2.circle(vis, pr, ring, col, thick)
        cv2.line(vis, pl, pr, col, 1)
        mid = ((pl[0]+pr[0])//2, (pl[1]+pr[1])//2 - 12)
        label = f"{name}:{dists[name]:.0f}px"
        if collapsed:
            label += " !!!"
        cv2.putText(vis, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    banner_h = 80
    cv2.rectangle(vis, (0, 0), (frame_w, banner_h), (20,20,20), -1)
    cv2.putText(vis, f"f={fidx:05d}  pattern={pattern}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,255), 2)
    cv2.putText(vis, f"thresh={thresh:.0f}px (5%w)",
                (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 2)
    return vis


def main():
    OUT_DIR.mkdir(exist_ok=True)
    caps_dir = OUT_DIR / "collapse_caps"
    caps_dir.mkdir(exist_ok=True)

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
    thresh  = frame_w * SAME_POS_FRAC

    print(f"대상: {VIDEO_PATH.name}  {frame_w}x{frame_h}  {total}f  {fps:.1f}fps")
    print(f"SamePos threshold: {thresh:.0f}px (=5% frame_w)")

    # 카운터
    cnt_total_pose  = 0
    cnt_samepos     = 0
    cnt_ankle_only  = 0   # hip·knee 정상, ankle만 붕괴
    cnt_knee_ankle  = 0   # hip 정상, knee+ankle 붕괴
    cnt_all_collapse= 0   # hip+knee+ankle 전부 붕괴
    cnt_other       = 0   # ankle 붕괴이지만 hip 붕괴, knee 정상 등 비정형

    csv_rows:   List[Dict] = []
    caps:       List       = []   # (fidx, pattern, img)

    # 전체 프레임용 거리 분포 (hip/knee/ankle)
    all_dists = {"hip": [], "knee": [], "ankle": []}

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms   = int(round(frame_index * 1000.0 / fps))
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_res  = detector.detect_for_video(mp_img, ts_ms)
        lms     = mp_res.pose_landmarks[0] if mp_res.pose_landmarks else None

        if lms is None:
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"  {frame_index}/{total}", flush=True)
            continue

        cnt_total_pose += 1

        # 각 관절 left-right 거리 계산
        dists = {}
        confs = {}
        for name, (li, ri) in JOINTS.items():
            lm_l = lms[li]; lm_r = lms[ri]
            pl   = (lm_l.x * frame_w, lm_l.y * frame_h)
            pr   = (lm_r.x * frame_w, lm_r.y * frame_h)
            dists[name] = _dist(pl, pr)
            confs[name] = (_conf(lm_l), _conf(lm_r))
            all_dists[name].append(dists[name])

        ankle_same = dists["ankle"] < thresh

        if not ankle_same:
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"  {frame_index}/{total}", flush=True)
            continue

        # ── SamePos 프레임 분석 ─────────────────────────────────────
        cnt_samepos += 1
        knee_same  = dists["knee"]  < thresh
        hip_same   = dists["hip"]   < thresh

        if   not hip_same and not knee_same:
            pattern = "ankle_only"
            cnt_ankle_only += 1
        elif not hip_same and knee_same:
            pattern = "knee+ankle"
            cnt_knee_ankle += 1
        elif hip_same and knee_same:
            pattern = "all_collapse"
            cnt_all_collapse += 1
        else:  # hip_same but not knee_same (비정형)
            pattern = "hip+ankle"
            cnt_other += 1

        # CSV 행
        lm_la = lms[27]; lm_ra = lms[28]
        lm_lk = lms[25]; lm_rk = lms[26]
        lm_lh = lms[23]; lm_rh = lms[24]
        csv_rows.append({
            "frame_idx":     frame_index,
            "ts_ms":         ts_ms,
            "pattern":       pattern,
            # 거리
            "hip_dist_px":   round(dists["hip"],   1),
            "knee_dist_px":  round(dists["knee"],  1),
            "ankle_dist_px": round(dists["ankle"], 1),
            # 발목 좌표
            "la_x": round(lm_la.x*frame_w, 1),
            "la_y": round(lm_la.y*frame_h, 1),
            "ra_x": round(lm_ra.x*frame_w, 1),
            "ra_y": round(lm_ra.y*frame_h, 1),
            # 무릎 좌표
            "lk_x": round(lm_lk.x*frame_w, 1),
            "lk_y": round(lm_lk.y*frame_h, 1),
            "rk_x": round(lm_rk.x*frame_w, 1),
            "rk_y": round(lm_rk.y*frame_h, 1),
            # hip 좌표
            "lh_x": round(lm_lh.x*frame_w, 1),
            "lh_y": round(lm_lh.y*frame_h, 1),
            "rh_x": round(lm_rh.x*frame_w, 1),
            "rh_y": round(lm_rh.y*frame_h, 1),
            # confidence
            "la_conf": round(confs["ankle"][0], 4),
            "ra_conf": round(confs["ankle"][1], 4),
            "lk_conf": round(confs["knee"][0],  4),
            "rk_conf": round(confs["knee"][1],  4),
            "lh_conf": round(confs["hip"][0],   4),
            "rh_conf": round(confs["hip"][1],   4),
        })

        # 캡처 (패턴별 최대 5개씩, 전체 15개)
        per_pattern_counts = {
            "ankle_only": sum(1 for c in caps if c[1]=="ankle_only"),
            "knee+ankle":  sum(1 for c in caps if c[1]=="knee+ankle"),
            "all_collapse":sum(1 for c in caps if c[1]=="all_collapse"),
            "hip+ankle":   sum(1 for c in caps if c[1]=="hip+ankle"),
        }
        if per_pattern_counts.get(pattern, 0) < 5:
            vis = draw_collapse(frame, lms, frame_w, frame_h,
                                thresh, dists, pattern, frame_index)
            caps.append((frame_index, pattern, vis))

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  {frame_index}/{total}", flush=True)

    cap.release()
    detector.close()

    # ── CSV 저장 ────────────────────────────────────────────────────
    if csv_rows:
        cp = OUT_DIR / "collapse_frames.csv"
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV: {cp}  ({len(csv_rows)}행)")

    # ── 캡처 저장 ───────────────────────────────────────────────────
    for fidx, pat, img in caps:
        cv2.imwrite(str(caps_dir / f"f{fidx:05d}_{pat}.jpg"), img)
    print(f"캡처: {caps_dir}  ({len(caps)}개)")

    # ── 전체 관절 거리 분포 (참고) ────────────────────────────────
    def stats(vals):
        if not vals: return "—"
        return (f"avg={sum(vals)/len(vals):.0f}px  "
                f"min={min(vals):.0f}px  "
                f"med={sorted(vals)[len(vals)//2]:.0f}px  "
                f"max={max(vals):.0f}px")

    # ── 통계 출력 ────────────────────────────────────────────────────
    pct = lambda n, d: f"{100*n/d:.1f}%" if d else "—"

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  랜드마크 붕괴 패턴 분석  |  {VIDEO_PATH.name}")
    lines.append(f"{'='*60}")
    lines.append(f"  포즈 감지 총 프레임:  {cnt_total_pose}")
    lines.append(f"  SamePos 프레임:       {cnt_samepos}  "
                 f"({pct(cnt_samepos, cnt_total_pose)})")
    lines.append(f"  threshold:            {thresh:.0f}px (5% frame_w)")
    lines.append(f"{'─'*60}")
    lines.append(f"  [붕괴 패턴 분류]  (기준: 각 관절 L-R 거리 < {thresh:.0f}px)")
    lines.append(f"  1. ankle만 붕괴         (hip·knee 정상):"
                 f"  {cnt_ankle_only:4d}  ({pct(cnt_ankle_only,  cnt_samepos)})")
    lines.append(f"  2. knee+ankle 붕괴      (hip 정상):      "
                 f"  {cnt_knee_ankle:4d}  ({pct(cnt_knee_ankle,  cnt_samepos)})")
    lines.append(f"  3. hip+knee+ankle 전부: "
                 f"  {cnt_all_collapse:4d}  ({pct(cnt_all_collapse,cnt_samepos)})")
    lines.append(f"  4. 비정형 (hip+ankle):  "
                 f"  {cnt_other:4d}  ({pct(cnt_other,        cnt_samepos)})")
    lines.append(f"{'─'*60}")
    lines.append(f"  [전체 프레임 L-R 관절 거리 분포]")
    lines.append(f"  hip   : {stats(all_dists['hip'])}")
    lines.append(f"  knee  : {stats(all_dists['knee'])}")
    lines.append(f"  ankle : {stats(all_dists['ankle'])}")
    lines.append(f"{'─'*60}")
    lines.append(f"  [SamePos 프레임 관절별 평균 거리]")
    if csv_rows:
        avg = lambda key: sum(r[key] for r in csv_rows)/len(csv_rows)
        lines.append(f"  hip   avg: {avg('hip_dist_px'):.1f}px")
        lines.append(f"  knee  avg: {avg('knee_dist_px'):.1f}px")
        lines.append(f"  ankle avg: {avg('ankle_dist_px'):.1f}px")
        lines.append(f"")
        lines.append(f"  [결론 힌트]")
        pct_ao = 100*cnt_ankle_only/cnt_samepos if cnt_samepos else 0
        if pct_ao >= 60:
            lines.append(f"  → ankle만 붕괴 {pct_ao:.0f}%")
            lines.append(f"    hip·knee 는 안정적")
            lines.append(f"    = 발목 기반 벡터를 hip·knee 기반으로 교체 가능")
        elif (cnt_ankle_only + cnt_knee_ankle) / cnt_samepos >= 0.7:
            lines.append(f"  → knee+ankle 까지만 붕괴")
            lines.append(f"    hip 은 안정적")
            lines.append(f"    = hip 기반 벡터는 신뢰 가능")
        else:
            lines.append(f"  → hip·knee·ankle 전체 붕괴 비율 높음")
            lines.append(f"    = MediaPipe Pose 자체가 이 환경에 부적합")
    lines.append(f"{'='*60}")

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "collapse_summary.txt").write_text(summary+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
