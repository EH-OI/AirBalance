# proj.py
# AirBalance 통합 파이프라인
#   [왼쪽 분기] 초기화: 팀원 코드 (YOLO + InitializationManager) → 개인 임계각 산출
#   [오른쪽 분기] 모니터링: MediaPipe + AirBalanceMonitor → 위험도 판정 → 경고 출력

import argparse
import csv
import json
import math
import platform
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

# ── 팀원 모듈 ────────────────────────────────────
from video_source import VideoSourceManager
from yolo_processor import YOLOProcessor
from init_manager import InitializationManager
from visualizer import Visualizer
from constants import (
    InitState, COLOR_INIT_TEXT, INIT_DURATION,
    COLOR_HIP_ANKLE, COLOR_KNEE, COLOR_SKELETON, COLOR_LEG_LINE,
)

# ── 경로 상수 ────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "pose_landmarker.task"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# MediaPipe Pose 랜드마크 인덱스 (hip=23/24, knee=25/26, ankle=27/28)
LANDMARK_INDEX = {
    "left":  (23, 25, 27),
    "right": (24, 26, 28),
}

# 단계별 색상 (BGR) 및 영문 표시명
STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "정상": (45, 196, 76),
    "주의": (0, 220, 255),
    "위험": (20, 20, 235),
}
DISPLAY_STATE = {
    "정상": "NORMAL",
    "주의": "CAUTION",
    "위험": "DANGER",
}
STATE_ORDER: Dict[str, int] = {"정상": 0, "주의": 1, "위험": 2}

# 소리 알림 쿨다운 (초)
_ALERT_COOLDOWN = {"위험": 1.5}


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class SafetyProfile:
    """초기화 완료 후 생성되는 개인 안전 파라미터.

    baseline_deg : 개인 기준각 (상위 15% 프레임 최대각 평균)
    warning_deg  : 위험 판정 임계각 = baseline × DANG_MULT (기본 1.8)
    danger_deg   : warning_deg 와 동일 (하위 호환용 필드, baseline × DANG_MULT)
    """
    initialized: bool
    baseline_deg: float          # 개인 기준각 (정상 최대 스윙)
    warning_deg: float           # 위험 판정 임계각 (baseline × DANG_MULT, 1.8)
    danger_deg: float            # 위험 임계각 (baseline × 1.8)
    omega_limit_deg_s: float     # 각속도 한계 (deg/s)
    source: str                  # 파라미터 출처 설명
    leg_length_px: float = 0.0   # 개인 다리 길이 (픽셀, 사용자 교체 감지용)


@dataclass
class LegReading:
    """한 쪽 다리의 단일 프레임 관절 측정값."""
    side: str
    angle_deg: float            # hip-ankle 기울기 또는 hip-knee-ankle 내각 (deg)
    hip: Tuple[int, int]
    knee: Optional[Tuple[int, int]]
    ankle: Tuple[int, int]
    confidence: float
    is_extrapolated: bool = False   # True: 속도 외삽 추정값


@dataclass
class Measurement:
    """위험도 판정 결과."""
    side: str
    raw_theta_deg: float        # EMA 적용 전 원시 각도
    theta_deg: float            # EMA 스무딩 각도
    omega_deg_s: float          # EMA 스무딩 각속도
    score: float                # warning 대비 각도 비율 (0~150+, 표시용)
    average_score: float        # 경고 구간 지속 시간 (초, sustain 표시용)
    state: str                  # 정상/주의/위험
    color: Tuple[int, int, int]
    limit_exceeded: bool        # warning_deg 초과 여부


# ──────────────────────────────────────────────
# 다리별 모션 상태 (EMA 필터용)
# ──────────────────────────────────────────────

@dataclass
class LegMotionState:
    prev_time: Optional[float] = None
    prev_theta: Optional[float] = None
    filtered_theta: Optional[float] = None
    smoothed_omega: float = 0.0
    last_valid_time: Optional[float] = None
    score_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    caution_seconds: float = 0.0      # CAUTION 구간(≥caution_deg) 지속 시간 (초)
    danger_seconds:  float = 0.0      # DANGER 구간(≥danger_deg) 지속 시간 (초)


# ──────────────────────────────────────────────
# AirBalanceMonitor  (오른쪽 분기 핵심)
# ──────────────────────────────────────────────

class AirBalanceMonitor:
    """
    개인화 임계값 기반 실시간 위험 판정 (3단계: 정상/주의/위험).

    알고리즘:
      1. EMA 스무딩으로 각도·각속도 산출
      2. 순수 상대 임계값:
           caution = baseline × 1.2   (CAUTION 시작)
           danger  = baseline × 1.8   (DANGER 시작 — warning 각도 이상 즉시 위험)
      3. 레벨별 독립 지속 시간 조건:
           CAUTION: ≥caution_deg 상태 0.15s 이상 연속 → '주의'
           DANGER:  ≥danger_deg 상태 0.3s 이상 연속  → '위험'
           기준 이하로 내려가면 해당 레벨 타이머 리셋.

    분류 기준:
      danger_seconds  ≥ SUSTAIN_DANGER (0.3s)  → 위험
      caution_seconds ≥ SUSTAIN_CAUTION (0.15s) → 주의
      그 외                                      → 정상
    """

    # ── 개인화 배율 ─────────────────────────────────────────────────────────
    # baseline=10° → caution=12°, danger=18°  (학생)
    # baseline=20° → caution=24°, danger=36°  (성인)
    CAUTION_MULT: float = 1.2  # caution = baseline × CAUTION_MULT
    DANG_MULT:    float = 1.8  # danger  = baseline × DANG_MULT (구 warning 각도)

    # ── 레벨별 지속 시간 조건 (초) ───────────────────────────────────────────
    SUSTAIN_CAUTION: float = 0.15  # CAUTION: 한 발걸음 피크(~0.15s) 이상
    SUSTAIN_DANGER:  float = 0.3   # DANGER:  연속 2보 이상

    def __init__(
        self,
        profile: SafetyProfile,
        angle_weight: float = 0.85,        # 하위호환용 (미사용)
        persistence_seconds: float = 0.5,  # 하위호환용 (미사용)
        angle_smoothing_tau: float = 0.08,
        omega_smoothing_tau: float = 0.15,
    ) -> None:
        self.profile     = profile
        self.baseline    = profile.baseline_deg
        self.caution     = self.baseline * self.CAUTION_MULT  # 1.2× 초과 시 CAUTION
        self.danger      = profile.warning_deg                # warning 이상 → 즉시 DANGER
        self.omega_limit = profile.omega_limit_deg_s
        self.angle_tau   = angle_smoothing_tau
        self.omega_tau   = omega_smoothing_tau
        self._states: Dict[str, LegMotionState] = {
            "left": LegMotionState(), "right": LegMotionState(),
        }

    # ── 각도 계산 ────────────────────────────────

    @staticmethod
    def _inter_leg_angle(r0: "LegReading", r1: "LegReading") -> float:
        """두 다리 벡터(anchor→ankle) 사이의 벌어짐 각도 (0~180°)."""
        v0 = np.array([r0.ankle[0] - r0.hip[0], r0.ankle[1] - r0.hip[1]], dtype=float)
        v1 = np.array([r1.ankle[0] - r1.hip[0], r1.ankle[1] - r1.hip[1]], dtype=float)
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 < 1e-8 or n1 < 1e-8:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))))

    @staticmethod
    def _angle_3pt(
        hip: Tuple[float, float],
        knee: Tuple[float, float],
        ankle: Tuple[float, float],
    ) -> float:
        """무릎을 꼭짓점으로 하는 hip-knee-ankle 내각 (0~180°)."""
        v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]], dtype=float)
        v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

    @staticmethod
    def _angle_2pt(
        hip: Tuple[float, float],
        ankle: Tuple[float, float],
    ) -> float:
        """중력 벡터(아래 방향)와 hip→ankle 벡터 사이의 각도."""
        v = np.array([ankle[0] - hip[0], ankle[1] - hip[1]], dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(v[1] / norm, -1.0, 1.0))))

    @classmethod
    def calculate_angle(
        cls,
        hip: Tuple[float, float],
        ankle: Tuple[float, float],
        knee: Optional[Tuple[float, float]] = None,
    ) -> float:
        """3점 우선, knee 없으면 2점 폴백."""
        return cls._angle_3pt(hip, knee, ankle) if knee is not None else cls._angle_2pt(hip, ankle)

    # ── EMA ──────────────────────────────────────

    @staticmethod
    def _ema_alpha(dt: float, tau: float) -> float:
        return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)

    def _update_motion(
        self, raw_theta: float, ts: float, st: LegMotionState
    ) -> Tuple[float, float]:
        """EMA 스무딩 각도 및 각속도 반환."""
        if st.prev_time is None or st.filtered_theta is None:
            st.prev_time = ts
            st.prev_theta = raw_theta
            st.filtered_theta = raw_theta
            st.smoothed_omega = 0.0
            return raw_theta, 0.0

        dt = max(ts - st.prev_time, 1e-4)
        a = self._ema_alpha(dt, self.angle_tau)
        filtered = a * raw_theta + (1.0 - a) * st.filtered_theta

        raw_omega = abs(filtered - st.filtered_theta) / dt
        b = self._ema_alpha(dt, self.omega_tau)
        st.smoothed_omega = b * raw_omega + (1.0 - b) * st.smoothed_omega

        st.prev_time = ts
        st.prev_theta = raw_theta
        st.filtered_theta = filtered
        return filtered, st.smoothed_omega

    # ── 가려짐 처리 ──────────────────────────────

    def _handle_occlusion(self, ts: float, st: LegMotionState) -> None:
        """0.25초 이상 감지 공백이면 EMA + 지속 시간 초기화."""
        if st.last_valid_time is not None and ts - st.last_valid_time > 0.25:
            st.prev_time       = None
            st.prev_theta      = None
            st.filtered_theta  = None
            st.smoothed_omega  = 0.0
            st.caution_seconds = 0.0
            st.danger_seconds  = 0.0

    # ── 공개 진입점 ──────────────────────────────

    def process_readings(
        self, readings: List[LegReading], ts: float
    ) -> Optional[Measurement]:
        """
        양쪽 다리를 독립적으로 추적하고,
        더 위험한 쪽(STATE_ORDER 기준)의 Measurement 를 반환한다.
        """
        present_sides = {r.side for r in readings}
        for side, st in self._states.items():
            if side not in present_sides:
                self._handle_occlusion(ts, st)

        if not readings:
            return None

        candidates: List[Measurement] = []
        for r in readings:
            st = self._states[r.side]

            # dt: sustain 계산용 (prev_time 갱신 전에 먼저 확보)
            dt = max(ts - st.prev_time, 0.0) if st.prev_time is not None else 0.0

            # raw: 스파이크 클램프 후 임계값 비교 및 EMA 공용 입력
            raw = r.angle_deg
            # 랜드마크 오인식 방지: 이전 EMA 대비 40° 초과 점프는 직전 EMA 값으로 대체.
            # (MediaPipe가 한 프레임 랜드마크를 잘못 잡으면 raw=66° 같은 스파이크 발생 → EMA 폭주)
            if st.filtered_theta is not None and abs(raw - st.filtered_theta) > 40.0:
                raw = st.filtered_theta

            # EMA: 클램핑된 raw 사용 → 화면 표시·각속도 계산 (스파이크 전파 차단)
            theta, omega = self._update_motion(raw, ts, st)

            # ── 레벨별 독립 지속 시간 누적 (raw 기준) ──────────────────────
            # 임계값 이하로 내려가면 즉시 리셋이 아니라 감쇠(decay):
            #   - 진동하는 각도가 임계값 경계를 반복 통과할 때 타이머가 순간 리셋되는
            #     문제를 방지. caution 3배속 감쇠, danger 2배속 감쇠.
            if raw >= self.danger:
                st.danger_seconds  += dt
            else:
                st.danger_seconds   = max(0.0, st.danger_seconds - dt * 2.0)

            if raw >= self.caution:
                st.caution_seconds += dt
            else:
                st.caution_seconds  = max(0.0, st.caution_seconds - dt * 3.0)

            # ── 상태 분류: 정상 / 주의 / 위험 (3단계) ───────────────────────
            if st.danger_seconds >= self.SUSTAIN_DANGER:
                state = "위험"
            elif st.caution_seconds >= self.SUSTAIN_CAUTION:
                state = "주의"
            else:
                state = "정상"

            # score: danger 대비 비율 (표시 전용)
            score = min(150.0, raw / self.danger * 100.0) if self.danger > 0 else 0.0

            # over_s: 활성 레벨 지속 시간 (오버레이 표시용)
            over_s = st.danger_seconds if raw >= self.danger else st.caution_seconds

            st.last_valid_time = ts
            candidates.append(Measurement(
                side=r.side,
                raw_theta_deg=r.angle_deg,
                theta_deg=theta,
                omega_deg_s=omega,
                score=score,
                average_score=over_s,
                state=state,
                color=STATE_COLORS[state],
                limit_exceeded=(raw >= self.danger),
            ))

        return max(candidates, key=lambda m: STATE_ORDER[m.state])


# ──────────────────────────────────────────────
# Pose estimator  (MediaPipe Tasks)
# ──────────────────────────────────────────────

class PoseEstimator:
    """
    MediaPipe Tasks PoseLandmarker (VIDEO 모드).
    YOLO보다 세밀한 33점 랜드마크를 제공하며,
    측면 영상에서 hip/knee/ankle 가시성이 우수하다.
    """

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe 모델 파일 없음: {model_path}\n"
                "다운로드 명령어:\n"
                "  curl -o pose_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.30,
            min_pose_presence_confidence=0.30,
            min_tracking_confidence=0.30,
            output_segmentation_masks=False,
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> Optional[List[Any]]:
        """BGR 프레임을 받아 33개 NormalizedLandmark 리스트 반환. 미감지 시 None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def close(self) -> None:
        self.detector.close()


# ──────────────────────────────────────────────
# Frame clock  (타임스탬프 정확도 보장)
# ──────────────────────────────────────────────

class FrameClock:
    """웹캠 → 실시간 벽시계 / 저장 영상 → 영상 타임라인 타임스탬프 생성."""

    def __init__(self, live_input: bool, fps: float) -> None:
        self.live_input = live_input
        self.fps = fps if fps > 0 else 30.0
        self.started_at = time.perf_counter()
        self._prev_s = -1.0

    def next_seconds(self, cap: cv2.VideoCapture, frame_index: int) -> float:
        if self.live_input:
            s = time.perf_counter() - self.started_at
        else:
            reported = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            estimated = frame_index / self.fps
            s = max(reported, estimated)
        min_next = self._prev_s + 1.0 / self.fps
        s = max(s, min_next) if self._prev_s >= 0 else s
        self._prev_s = s
        return s


# ──────────────────────────────────────────────
# 랜드마크 → LegReading 변환
# ──────────────────────────────────────────────

def _lm_confidence(lm: Any) -> float:
    """visibility 와 presence 중 낮은 값을 신뢰도로 사용."""
    return min(
        float(getattr(lm, "visibility", 0.0)),
        float(getattr(lm, "presence", 0.0)),
    )


# ──────────────────────────────────────────────
# 랜드마크 좌표 EMA 스무더 (단순 버전)
# ──────────────────────────────────────────────

class LandmarkSmoother:
    """
    hip/knee/ankle 랜드마크 (x, y)에 시간 기반 EMA를 적용한다.
    각도 계산 전에 호출해 시각적 지터를 줄인다.
    """
    SMOOTH_IDX = {23, 24, 25, 26, 27, 28}  # left/right hip, knee, ankle

    def __init__(self, tau: float = 0.04) -> None:
        self.tau = tau
        self._prev: Dict[int, Tuple[float, float]] = {}
        self._prev_ts: Optional[float] = None

    def smooth(self, landmarks: List[Any], ts: float) -> List[Any]:
        if self._prev_ts is None or (ts - self._prev_ts) > 0.3:
            # 첫 프레임 또는 긴 공백 → 리셋
            self._prev.clear()
            for i, lm in enumerate(landmarks):
                if i in self.SMOOTH_IDX:
                    self._prev[i] = (lm.x, lm.y)
            self._prev_ts = ts
            return landmarks

        dt   = max(ts - self._prev_ts, 1e-4)
        alpha = 1.0 - math.exp(-dt / self.tau)
        self._prev_ts = ts

        result: List[Any] = []
        for i, lm in enumerate(landmarks):
            if i not in self.SMOOTH_IDX:
                result.append(lm)
                continue
            px, py = self._prev.get(i, (lm.x, lm.y))
            sx = alpha * lm.x + (1.0 - alpha) * px
            sy = alpha * lm.y + (1.0 - alpha) * py
            self._prev[i] = (sx, sy)

            class _S:
                pass
            s = _S()
            s.x = sx; s.y = sy; s.z = lm.z
            s.visibility = float(getattr(lm, "visibility", 0.0))
            s.presence   = float(getattr(lm, "presence",   0.0))
            result.append(s)
        return result


# ──────────────────────────────────────────────
# 3단계 하드 제약 필터
# ──────────────────────────────────────────────

class LegMotionFilter:
    """
    실험 C (Anti-Snap OFF + NMS OFF) 결과 반영:
    1-vector 16.3% → 0.7% 로 개선됨 (diag_ablation.py 계측).

    Anti-Snap 제거 근거: 정상 움직임도 108px 기준에 걸려 한쪽 다리 폐기 → 1-vector 유발.
    NMS 제거 근거: Anti-Snap 제거 시 NMS 발동이 오히려 85→163프레임 급증, 단독 효과 없음.

    현재 동작:
      1. 상태 업데이트 : 감지된 reading 의 velocity-EMA 갱신
      2. Extrapolation : 누락된 side 를 속도 기반 ghost vector 로 유지 (MAX_EXTRAP 프레임)
    """
    MAX_EXTRAP = 6     # 최대 연속 외삽 프레임 수
    VEL_ALPHA  = 0.45  # 속도 EMA 계수

    def __init__(self) -> None:
        self._st: Dict[str, Dict] = {
            side: {
                "ankle":   None,       # 직전 발목 픽셀 좌표
                "hip":     None,       # 직전 hip 픽셀 좌표
                "vel":     (0.0, 0.0), # 발목 이동 속도 (px/frame, EMA)
                "missing": 0,          # 연속 누락 프레임 수
            }
            for side in ("left", "right")
        }

    def process(
        self,
        readings: List[LegReading],
        frame_h: int,
        frame_w: int,
    ) -> List[LegReading]:
        # ── 1. 상태 업데이트 (Anti-Snap·NMS 없이 모든 reading 수용) ──
        for r in readings:
            st = self._st[r.side]
            if st["ankle"] is not None:
                raw_vx = float(r.ankle[0] - st["ankle"][0])
                raw_vy = float(r.ankle[1] - st["ankle"][1])
                pvx, pvy = st["vel"]
                st["vel"] = (
                    self.VEL_ALPHA * raw_vx + (1 - self.VEL_ALPHA) * pvx,
                    self.VEL_ALPHA * raw_vy + (1 - self.VEL_ALPHA) * pvy,
                )
            st["ankle"]   = r.ankle
            st["hip"]     = r.hip
            st["missing"] = 0

        # ── 2. Extrapolation: 누락된 side → ghost vector 유지 ────────
        present_sides = {r.side for r in readings}
        result: List[LegReading] = list(readings)

        for side in ("left", "right"):
            if side in present_sides:
                continue
            st = self._st[side]
            if st["ankle"] is None or st["missing"] >= self.MAX_EXTRAP:
                st["missing"] += 1
                continue

            vx, vy = st["vel"]
            new_ankle: Tuple[int,int] = (
                int(np.clip(st["ankle"][0] + vx, 0, frame_w - 1)),
                int(np.clip(st["ankle"][1] + vy, 0, frame_h - 1)),
            )
            if st["hip"] is not None:
                new_hip: Tuple[int,int] = (
                    int(np.clip(st["hip"][0] + vx * 0.3, 0, frame_w - 1)),
                    int(np.clip(st["hip"][1] + vy * 0.3, 0, frame_h - 1)),
                )
            else:
                new_hip = new_ankle

            angle = AirBalanceMonitor._angle_2pt(
                (float(new_hip[0]), float(new_hip[1])),
                (float(new_ankle[0]), float(new_ankle[1])),
            )
            result.append(LegReading(
                side=side, angle_deg=angle,
                hip=new_hip, knee=None, ankle=new_ankle,
                confidence=0.0, is_extrapolated=True,
            ))
            st["missing"] += 1

        return result


def extract_leg_readings(
    landmarks: Optional[List[Any]],
    frame: np.ndarray,
    side_option: str,
    vis_threshold: float,
    force_2pt: bool = True,
) -> Tuple[List[LegReading], Optional[float]]:
    """
    MediaPipe 랜드마크에서 LegReading 리스트를 추출한다.

    항상 좌/우 2개의 reading을 생성하는 것을 목표로 한다.
    - 신뢰도 임계값을 낮게 유지해 양쪽 다리가 동시에 감지되도록 한다.
    - dedup 없음: 두 개의 reading은 항상 유지된다.
    - 기구 봉 오인식 방지: 뒤집힘 체크 + 최소 세그먼트 길이만 적용.
    """
    if landmarks is None:
        return [], None

    h, w = frame.shape[:2]
    sides: Iterable[str] = ("left", "right") if side_option == "both" else (side_option,)
    readings: List[LegReading] = []

    for side in sides:
        hip_idx, knee_idx, ankle_idx = LANDMARK_INDEX[side]
        lm_hip   = landmarks[hip_idx]
        lm_knee  = landmarks[knee_idx]
        lm_ankle = landmarks[ankle_idx]

        knee_conf  = _lm_confidence(lm_knee)
        ankle_conf = _lm_confidence(lm_ankle)
        hip_conf   = _lm_confidence(lm_hip)

        hip_f   = (lm_hip.x * w,   lm_hip.y * h)
        knee_f  = (lm_knee.x * w,  lm_knee.y * h)
        ankle_f = (lm_ankle.x * w, lm_ankle.y * h)

        knee_ok = knee_conf >= 0.05   # 아주 낮은 최소치만 확인

        # ── 기본 기하 필터: hip→ankle 기반 (init_manager와 동일 스케일) ─────
        # init_manager는 hip→ankle 각도로 baseline을 산출하므로,
        # 모니터링도 hip→ankle 각도를 사용해야 임계값 비교가 일관된다.
        # (knee→ankle을 쓰면 동일 보폭이어도 측정값이 더 작게 나와 임계값이 걸리지 않음)
        seg_len = float(np.linalg.norm(np.array(ankle_f) - np.array(hip_f)))
        if seg_len < h * 0.05:                     # 5% 미만 → 점 수준
            continue
        if hip_f[1] > ankle_f[1] + h * 0.08:      # hip이 ankle보다 아래 → 뒤집힘
            continue
        angle = AirBalanceMonitor._angle_2pt(hip_f, ankle_f)

        # 시각화: hip→knee→ankle 풀 스켈레톤
        readings.append(LegReading(
            side=side,
            angle_deg=angle,
            hip=(int(hip_f[0]),   int(hip_f[1])),
            knee=(int(knee_f[0]), int(knee_f[1])) if knee_ok else None,
            ankle=(int(ankle_f[0]), int(ankle_f[1])),
            confidence=min(hip_conf, ankle_conf),
        ))

    # ── spread 계산 (dedup 없음 — 항상 2개 유지) ─────────────────────────
    spread_angle: Optional[float] = None
    if len(readings) == 2:
        spread_angle = AirBalanceMonitor._inter_leg_angle(readings[0], readings[1])

    return readings, spread_angle


# ──────────────────────────────────────────────
# 경고 출력  (시각 + 청각)
# ──────────────────────────────────────────────

_alert_lock = threading.Lock()
_last_alert_time: Dict[str, float] = {"위험": 0.0}


def _play_sound(state: str) -> None:
    """플랫폼별 시스템 사운드 비동기 재생."""
    sounds = {
        "위험": {
            "Darwin":  "/System/Library/Sounds/Basso.aiff",
            "Linux":   "bell",
        },
    }
    sys_p = platform.system()
    try:
        if sys_p == "Darwin":
            path = sounds[state]["Darwin"]
            subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif sys_p == "Linux":
            subprocess.Popen(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif sys_p == "Windows":
            import winsound
            freq = 800 if state == "경고" else 1200
            winsound.Beep(freq, 300)
    except Exception:
        pass


def trigger_alert(state: str) -> None:
    """쿨다운을 지키면서 소리 알림을 비동기 실행한다."""
    if state not in _ALERT_COOLDOWN:
        return
    with _alert_lock:
        now = time.time()
        if now - _last_alert_time[state] < _ALERT_COOLDOWN[state]:
            return
        _last_alert_time[state] = now
    threading.Thread(target=_play_sound, args=(state,), daemon=True).start()


# ──────────────────────────────────────────────
# 모니터링 오버레이 렌더링
# ──────────────────────────────────────────────

def draw_monitoring_overlay(
    frame: np.ndarray,
    readings: List[LegReading],
    measurement: Optional[Measurement],
    profile: SafetyProfile,
    spread_angle: Optional[float] = None,
    spread_limit: float = 90.0,
    tracking_ok: bool = True,
) -> None:
    """팀원 visualizer 스타일 스켈레톤 + 반투명 패널 + 위험 경계선 그리기."""

    # ── 1. 스켈레톤 (example1 스타일) ──────────────────────────────────────
    skel_alpha   = 1.0 if tracking_ok else 0.45
    overlay_skel = frame.copy()

    for r in readings:
        active     = measurement is not None and r.side == measurement.side
        ghost      = not tracking_ok
        extrap     = r.is_extrapolated

        if extrap:
            # 외삽(추정) 벡터: 밝은 회색 + 중공 원으로 구별
            skel_col   = (160, 160, 160)
            meas_col   = (160, 160, 160)
            pt_col     = (160, 160, 160)
            line_thick = 2
        elif ghost:
            skel_col   = (120, 120, 120)
            meas_col   = (100, 100, 100)
            pt_col     = (120, 120, 120)
            line_thick = 2
        else:
            skel_col   = COLOR_SKELETON
            meas_col   = measurement.color if active else (160, 160, 160)
            pt_col     = COLOR_HIP_ANKLE
            line_thick = 4 if active else 2

        # 흰색 가이드선 (hip→knee→ankle 또는 hip→ankle)
        if r.knee is not None:
            cv2.line(overlay_skel, r.hip,  r.knee,  skel_col, 2)
            cv2.line(overlay_skel, r.knee, r.ankle, skel_col, 2)
        else:
            cv2.line(overlay_skel, r.hip,  r.ankle, skel_col, 2)

        # 굵은 측정 기준선
        meas_top = r.knee if r.knee is not None else r.hip
        cv2.line(overlay_skel, meas_top, r.ankle, meas_col, line_thick)

        # 관절 점
        cv2.circle(overlay_skel, r.hip,   6, pt_col, -1)
        cv2.circle(overlay_skel, r.ankle, 6, pt_col, -1)
        if r.knee is not None:
            cv2.circle(overlay_skel, r.knee, 6,
                       (60, 120, 60) if not (ghost or extrap) else pt_col, -1)

        # 외삽 표시: 발목에 중공 원
        if extrap:
            cv2.circle(overlay_skel, r.ankle, 10, pt_col, 1)

        # active 다리에 px 거리 표시
        if active and not ghost and not extrap:
            seg_px = int(np.linalg.norm(np.array(r.ankle) - np.array(meas_top)))
            mid = ((meas_top[0] + r.ankle[0]) // 2 + 12,
                   (meas_top[1] + r.ankle[1]) // 2)
            cv2.putText(overlay_skel, f"{seg_px}px", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HIP_ANKLE, 2)

    # 좌우 hip 연결선
    if len(readings) == 2:
        cv2.line(overlay_skel, readings[0].hip, readings[1].hip,
                 (120, 120, 120) if not tracking_ok else COLOR_SKELETON, 2)

    cv2.addWeighted(overlay_skel, skel_alpha, frame, 1.0 - skel_alpha, 0.0, frame)

    # ── 2. 반투명 정보 패널 ─────────────────────────────────────────────────
    spread_exceeded = spread_angle is not None and spread_angle > spread_limit
    panel_h = 244 if measurement is not None else 85
    panel_ov = frame.copy()
    cv2.rectangle(panel_ov, (8, 8), (370, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(panel_ov, 0.65, frame, 0.35, 0.0, frame)

    if measurement is None:
        cv2.putText(frame, "POSE NOT RELIABLE", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 180, 255), 2)
        return

    # ── 3. 수치 텍스트 ──────────────────────────────────────────────────────
    spread_str   = f"{spread_angle:5.1f} deg" if spread_angle is not None else "  --"
    spread_color = STATE_COLORS["위험"] if spread_exceeded else (200, 200, 200)
    data_color   = (160, 160, 160) if not tracking_ok else (245, 245, 245)
    state_txt    = (DISPLAY_STATE[measurement.state] + " [LOST]") if not tracking_ok \
                   else DISPLAY_STATE[measurement.state]
    over_s       = measurement.average_score   # 활성 레벨 지속 시간 (초)
    sustain_ref  = {
        "위험": AirBalanceMonitor.SUSTAIN_DANGER,
        "주의": AirBalanceMonitor.SUSTAIN_CAUTION,
        "정상": AirBalanceMonitor.SUSTAIN_CAUTION,
    }[measurement.state]
    text_lines = [
        (f"Leg   : {measurement.side.upper()}",                                             data_color),
        (f"Angle : {measurement.theta_deg:5.1f} / {profile.warning_deg:.1f} deg (danger)", data_color),
        (f"Vel.  : {measurement.omega_deg_s:5.1f} deg/s",                                  data_color),
        (f"Spread: {spread_str}",                                                            spread_color),
        (f"Over  : {over_s:5.2f}s / {sustain_ref:.1f}s",                                   measurement.color),
        (f"State : {state_txt}",                                                             measurement.color),
    ]
    for i, (line, color) in enumerate(text_lines):
        cv2.putText(frame, line, (20, 42 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2)

    # ── 4. 위험/벌어짐 경계 강조 ────────────────────────────────────────────
    bottom_y = frame.shape[0]
    if spread_exceeded:
        cv2.putText(frame, f"SPREAD WARNING: {spread_angle:.0f} deg",
                    (22, bottom_y - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, STATE_COLORS["위험"], 3)
    if measurement.state == "위험" and tracking_ok:
        cv2.rectangle(frame, (4, 4),
                      (frame.shape[1] - 4, frame.shape[0] - 4),
                      measurement.color, 14)
        cv2.putText(
            frame, "STOP! UNSAFE SWING DETECTED",
            (22, bottom_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, measurement.color, 3,
        )


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def find_default_video() -> Optional[Path]:
    """현재 폴더에서 첫 번째 영상 파일을 자동 탐색한다."""
    for ext in VIDEO_EXTENSIONS:
        candidates = sorted(SCRIPT_DIR.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def _derive_omega_limit(theta_limit: float) -> float:
    """
    theta_limit 에서 omega_limit 를 선형 파생한다.
    θ = 30° → ω ≈ 90 deg/s, θ = 60° → ω = 180 deg/s.
    범위 [60, 300] 으로 클리핑.
    """
    return float(np.clip(theta_limit * 3.0, 60.0, 300.0))


def scan_cameras(max_index: int = 6) -> List[Tuple[int, str]]:
    """
    사용 가능한 카메라 장치를 인덱스 0부터 순서대로 스캔한다.
    반환: [(인덱스, "WxH 해상도"), ...]
    """
    available: List[Tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"{w}x{h}"))
            cap.release()
    return available


def find_iphone_camera_index() -> Optional[int]:
    """
    아이폰 카메라 장치 인덱스를 자동 탐색한다.

    두 가지 연결 방식을 모두 지원한다:
      - Continuity Camera (연속성 카메라): macOS 13+ 내장 기능, 앱 불필요.
        같은 Apple ID + Wi-Fi/Bluetooth 활성화 시 자동 연결.
      - iVCam v7.x: 서드파티 앱 사용 시.

    macOS: system_profiler SPCameraDataType 으로 'iphone' 또는 'ivcam'
           문자열 확인 후 인덱스 1부터 스캔.
    탐색 실패 시 None 반환.
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=5,
            )
            info = result.stdout.lower()
            found = "iphone" in info or "ivcam" in info
            if not found:
                print("[iPhone 카메라] 시스템에서 아이폰 카메라를 찾을 수 없습니다.")
                print("  [Continuity Camera] 아이폰과 맥북이 같은 Apple ID에 로그인되어 있고")
                print("  Wi-Fi + Bluetooth 가 켜져 있는지 확인하세요. (macOS 13+, iOS 16+)")
                print("  [iVCam] iVCam v7 앱이 아이폰과 맥북 양쪽에서 실행 중인지 확인하세요.")
                return None
        except Exception:
            pass  # system_profiler 실패 시 스캔으로 폴백

    # 인덱스 0은 내장 카메라이므로 1부터 탐색
    for idx in range(1, 7):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return idx
    return None


# ──────────────────────────────────────────────
# 통합 파이프라인
# ──────────────────────────────────────────────

def run_integrated(args: argparse.Namespace) -> None:
    """
    통합 파이프라인 (기본 실행 모드).

    단계 1 [초기화 / 왼쪽 분기]:
        VideoSourceManager + YOLOProcessor + InitializationManager + Visualizer
        → 7초 데이터 수집 → 개인 안전 임계각 산출

    단계 2 [모니터링 / 오른쪽 분기]:
        PoseEstimator(MediaPipe) + AirBalanceMonitor
        → 관절 각도 산출 → 각속도 산출 → 위험도 점수 산출
        → 단계 분류 (정상/주의/위험) → 경고 출력 (시각/청각)
    """

    # ── 영상 소스 결정 ────────────────────────
    if args.webcam:
        source: Union[int, str] = args.camera_index
        live_input = True
    else:
        video_path = args.input or find_default_video()
        if video_path is None:
            source = args.camera_index
            live_input = True
        else:
            source = str(video_path)
            live_input = False

    # ── 팀원 모듈 초기화 ──────────────────────
    video = VideoSourceManager(source)
    video.open()
    yolo = YOLOProcessor()
    yolo.load_model()
    init_manager = InitializationManager()
    team_vis = Visualizer()

    # ── 내 모듈 초기화 ────────────────────────
    estimator = PoseEstimator(args.model)
    monitor: Optional[AirBalanceMonitor] = None
    profile: Optional[SafetyProfile] = None

    cap = video.cap  # VideoSourceManager 내부 cv2.VideoCapture
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_clock = FrameClock(live_input=live_input, fps=fps)

    # ── CSV 설정 ──────────────────────────────
    csv_file = csv_writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "timestamp_s", "phase",
            "side", "angle_mode",
            "theta_deg", "omega_deg_s",
            "score", "average_score", "state", "spread_deg",
            "n_real_legs",
            "l_hip_conf", "l_knee_conf", "l_ankle_conf",
            "r_hip_conf", "r_knee_conf", "r_ankle_conf",
        ])

    # ── 영상 저장 설정 ────────────────────────
    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_px, h_px)
        )

    print(f"[AirBalance] 입력: {source}")
    print(f"[AirBalance] MediaPipe 모델: {args.model}")
    if not args.no_display:
        print("[AirBalance] 종료: 창에서 q 또는 ESC")

    # ── 벤치마크 변수 ─────────────────────────────
    benchmark = getattr(args, 'benchmark', False)
    _t_yolo:      List[float] = []   # YOLO 추론 시간 (초기화 분기)
    _t_mediapipe: List[float] = []   # MediaPipe 추론 시간 (모니터링 분기)
    _t_frame:     List[float] = []   # 프레임 전체 처리 시간

    # ── 외삽 통계 카운터 ──────────────────────────
    _cnt_monitoring = 0   # 모니터링 총 프레임
    _cnt_both_real  = 0   # 양측 실측 감지
    _cnt_one_extrap = 0   # 단측 외삽 보완
    _cnt_fail       = 0   # 감지 완전 실패 (양측 누락)

    # ── confidence 추적 ───────────────────────────
    _conf_keys = ["l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
    _conf_data: Dict[str, List[float]] = {k: [] for k in _conf_keys}

    frame_index = 0
    try:
        while True:
            _t0 = time.perf_counter()

            ret, frame = video.read_frame()
            if not ret:
                if not getattr(args, 'loop', False):
                    break
                # 영상 위치만 처음으로 되감기.
                # frame_index·frame_clock·monitor·init_manager 는 그대로 유지:
                #   - 타임스탬프가 단조 증가해야 MediaPipe가 정상 동작
                #   - 초기화가 완료됐으면 측정 상태를 이어서 계속 사용
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            ts = frame_clock.next_seconds(cap, frame_index)

            # ════════════════════════════════════
            # [단계 1] 초기화 플로우 (왼쪽 분기)
            # ════════════════════════════════════
            if not init_manager.is_done():
                _ty0 = time.perf_counter()
                detection = yolo.detect(frame)
                _t_yolo.append(time.perf_counter() - _ty0)
                init_manager.process_frame(detection)

                if (detection is not None
                        and init_manager.state == InitState.IN_PROGRESS):
                    elapsed  = time.time() - init_manager.init_start_time
                    progress = min(1.0, elapsed / INIT_DURATION)
                    l_avg = (init_manager.landmark_buffer[-1]
                             if init_manager.landmark_buffer else 0.0)
                    team_vis.draw_init_overlay(
                        frame, detection["keypoints"],
                        progress, init_manager.get_state(), l_avg,
                        bbox=detection.get("bbox"),
                        display_slots=init_manager.get_display_slots(),
                        debug_info=init_manager.get_debug_info(),
                    )
                else:
                    cv2.putText(
                        frame, "Waiting for person...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_INIT_TEXT, 2,
                    )

            # ════════════════════════════════════
            # [단계 2] 모니터링 플로우 (오른쪽 분기)
            # ════════════════════════════════════
            else:
                # 초기화 완료 직후 모니터 1회 생성
                if monitor is None:
                    baseline    = init_manager.get_threshold() or args.theta_limit
                    dang_mult   = getattr(args, 'dang_mult', AirBalanceMonitor.DANG_MULT)
                    warning_deg = baseline * dang_mult
                    danger_deg  = warning_deg
                    omega_limit = _derive_omega_limit(baseline)
                    profile = SafetyProfile(
                        initialized=True,
                        baseline_deg=baseline,
                        warning_deg=warning_deg,
                        danger_deg=danger_deg,
                        omega_limit_deg_s=omega_limit,
                        source="InitializationManager",
                        leg_length_px=init_manager.leg_length_px or 0.0,
                    )
                    monitor = AirBalanceMonitor(profile=profile)
                    lm_smoother = LandmarkSmoother(tau=0.04)
                    leg_filter  = LegMotionFilter()
                    last_readings_i:    List[LegReading]         = []
                    last_measurement_i: Optional[Measurement]    = None
                    print(
                        f"[모니터링 시작] 기준각: {baseline:.1f}°  "
                        f"경고: {warning_deg:.1f}°  위험: {danger_deg:.1f}°  "
                        f"각속도: {omega_limit:.1f} deg/s"
                    )
                    # profile_saved.json 저장 (다음 실행 시 --profile 로 재사용)
                    import json as _json
                    _saved = {
                        "initialized": True,
                        "baseline_deg":       round(baseline, 3),
                        "warning_deg":        round(warning_deg, 3),
                        "danger_deg":         round(danger_deg, 3),
                        "omega_limit_deg_s":  round(omega_limit, 3),
                        "source":             str(source),
                        "leg_length_px":      round(profile.leg_length_px, 1),
                    }
                    try:
                        (SCRIPT_DIR / "profile_saved.json").write_text(
                            _json.dumps(_saved, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        print(f"[프로필 저장] profile_saved.json")
                    except Exception as _e:
                        print(f"[프로필 저장 실패] {_e}")

                    # --create-profile: 프로필 저장 후 즉시 종료 (모니터링 불필요)
                    if getattr(args, 'create_profile', False):
                        print(f"[--create-profile] 완료. profile_saved.json 을 --profile 로 재사용하세요.")
                        return

                # MediaPipe 포즈 추정
                ts_ms = int(round(ts * 1000.0))
                _tm0 = time.perf_counter()
                raw_lm = estimator.detect(frame, ts_ms)
                _t_mediapipe.append(time.perf_counter() - _tm0)
                landmarks = lm_smoother.smooth(raw_lm, ts) if raw_lm is not None else None

                if raw_lm is not None:
                    _fc = (
                        _lm_confidence(raw_lm[23]), _lm_confidence(raw_lm[25]), _lm_confidence(raw_lm[27]),
                        _lm_confidence(raw_lm[24]), _lm_confidence(raw_lm[26]), _lm_confidence(raw_lm[28]),
                    )
                else:
                    _fc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                for _ck, _cv in zip(_conf_keys, _fc):
                    _conf_data[_ck].append(_cv)

                # --debug-lm: 33개 랜드마크 전체 시각화 (진단용)
                if getattr(args, 'debug_lm', False) and landmarks is not None:
                    h_dbg, w_dbg = frame.shape[:2]
                    key_idx = {23: 'LH', 24: 'RH', 25: 'LK', 26: 'RK', 27: 'LA', 28: 'RA'}
                    for i, lm in enumerate(landmarks):
                        px = int(lm.x * w_dbg); py = int(lm.y * h_dbg)
                        vis = min(float(getattr(lm, 'visibility', 0)), float(getattr(lm, 'presence', 0)))
                        if vis < 0.1:
                            continue
                        color = (0, 255, 255) if i in key_idx else (180, 180, 180)
                        cv2.circle(frame, (px, py), 6 if i in key_idx else 3, color, -1)
                        if i in key_idx:
                            cv2.putText(frame, key_idx[i], (px + 6, py - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # 관절 각도 산출 + 3단계 필터 + 위험도 판정
                h_fr, w_fr = frame.shape[:2]
                raw_readings, spread_angle = extract_leg_readings(
                    landmarks, frame,
                    args.side, args.visibility_threshold,
                    force_2pt=(args.angle_mode == "2pt"),
                )
                readings = leg_filter.process(raw_readings, h_fr, w_fr)
                # spread 재계산 (extrapolation 포함 2개일 때)
                real = [r for r in readings if not r.is_extrapolated]
                _cnt_monitoring += 1
                n_real = len(real)
                if n_real >= 2:
                    _cnt_both_real += 1
                elif n_real == 1:
                    _cnt_one_extrap += 1
                else:
                    _cnt_fail += 1
                if len(real) == 2:
                    spread_angle = AirBalanceMonitor._inter_leg_angle(real[0], real[1])
                # 외삽 reading은 angle 계산에서 제외
                measurement = monitor.process_readings(real if real else readings, ts)

                # 다리 벌어짐 초과 시 최소 위험 상태로 강제 전환
                spread_limit = profile.warning_deg * 2.0
                if (spread_angle is not None
                        and spread_angle > spread_limit
                        and measurement is not None
                        and STATE_ORDER[measurement.state] < STATE_ORDER["위험"]):
                    measurement.state = "위험"
                    measurement.color = STATE_COLORS["위험"]

                # 감지 공백 시 마지막 유효 포즈를 ghosted로 표시
                tracking_ok_i = bool(readings)
                disp_r = readings    if readings    else last_readings_i
                disp_m = measurement if measurement is not None else last_measurement_i
                if readings:               last_readings_i    = readings
                if measurement is not None: last_measurement_i = measurement

                # 시각적 오버레이
                draw_monitoring_overlay(
                    frame, disp_r, disp_m, profile,
                    spread_angle=spread_angle, spread_limit=spread_limit,
                    tracking_ok=tracking_ok_i,
                )

                # 청각 알림
                if measurement is not None:
                    trigger_alert(measurement.state)

                # CSV 기록
                if csv_writer is not None and measurement is not None:
                    used = next(
                        (r for r in readings if r.side == measurement.side), None
                    )
                    mode = "3pt" if (used and used.knee) else "2pt"
                    csv_writer.writerow([
                        frame_index, f"{ts:.4f}", "monitoring",
                        measurement.side, mode,
                        f"{measurement.theta_deg:.3f}",
                        f"{measurement.omega_deg_s:.3f}",
                        f"{measurement.score:.3f}",
                        f"{measurement.average_score:.3f}",
                        measurement.state,
                        f"{spread_angle:.1f}" if spread_angle is not None else "",
                        n_real,
                        *[f"{v:.3f}" for v in _fc],
                    ])

            # ── 프레임 처리 시간 기록 ──────────────────
            _t_frame.append(time.perf_counter() - _t0)

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                delay = 1 if live_input else max(1, int(round(1000.0 / fps)))
                cv2.imshow("AirBalance Monitor", frame)
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    finally:
        estimator.close()
        video.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    print(f"[AirBalance] 완료: {frame_index} 프레임 처리")

    # ── 가려짐 보완 성능 통계 ─────────────────────────────────────────────────
    if _cnt_monitoring > 0:
        t = _cnt_monitoring
        print("\n┌───────────────────────────────┬────────┬─────────┐")
        print("│           가려짐 보완 성능 통계                   │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print("│ 구분                          │ 프레임 │  비율   │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print(f"│ 양측 실측 감지                │ {_cnt_both_real:6d} │ {_cnt_both_real/t*100:6.1f}% │")
        print(f"│ 단측 외삽 보완                │ {_cnt_one_extrap:6d} │ {_cnt_one_extrap/t*100:6.1f}% │")
        print(f"│ 감지 완전 실패 (양측 누락)    │ {_cnt_fail:6d} │ {_cnt_fail/t*100:6.1f}% │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print(f"│ 모니터링 총 프레임            │ {t:6d} │  100.0% │")
        print("└───────────────────────────────┴────────┴─────────┘")

    # ── confidence 통계 ───────────────────────────────────────────────────────
    if _conf_data["l_hip"]:
        labels = ["L Hip  ", "L Knee ", "L Ankle", "R Hip  ", "R Knee ", "R Ankle"]
        print("\n┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┐")
        print("│              랜드마크 confidence 통계                          │")
        print("├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤")
        print("│ 관절     │  mean  │  min   │  p25   │  p50   │  p75   │  max   │")
        print("├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤")
        for label, key in zip(labels, _conf_keys):
            arr = np.array(_conf_data[key])
            print(f"│ {label} │ {arr.mean():.3f}  │ {arr.min():.3f}  │ "
                  f"{np.percentile(arr,25):.3f}  │ {np.percentile(arr,50):.3f}  │ "
                  f"{np.percentile(arr,75):.3f}  │ {arr.max():.3f}  │")
        print("└──────────┴────────┴────────┴────────┴────────┴────────┴────────┘")

    # ── 벤치마크 요약 출력 ────────────────────────────────────────────────────
    if benchmark and _t_frame:
        def _ms(lst: List[float]) -> str:
            if not lst:
                return "N/A"
            avg = sum(lst) / len(lst) * 1000
            mx  = max(lst) * 1000
            return f"avg {avg:.1f}ms  max {mx:.1f}ms"

        total_s = sum(_t_frame)
        avg_fps = len(_t_frame) / total_s if total_s > 0 else 0.0
        print("\n┌─────────────────────────────────────────────┐")
        print("│              Benchmark 결과                 │")
        print("├─────────────────────────────────────────────┤")
        print(f"│ 총 프레임          : {frame_index:>6} frames            │")
        print(f"│ 평균 FPS           : {avg_fps:>6.1f} fps              │")
        print(f"│ 프레임 처리 시간   : {_ms(_t_frame):<28} │")
        print(f"│ YOLO 추론 시간     : {_ms(_t_yolo):<28} │")
        print(f"│ MediaPipe 추론 시간: {_ms(_t_mediapipe):<28} │")
        print("└─────────────────────────────────────────────┘")


# ──────────────────────────────────────────────
# 독립 모니터링 파이프라인 (--profile 지정 시)
# ──────────────────────────────────────────────

def run_standalone(args: argparse.Namespace) -> None:
    """
    초기화 단계를 건너뛰고 JSON 프로필로 바로 모니터링.
    개발·디버깅 또는 임계값이 미리 알려진 경우 사용.
    """
    with args.profile.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("initialized", False):
        raise ValueError("초기화가 완료되지 않은 프로필입니다.")

    # 새 형식(baseline/warning/danger) 또는 구 형식(theta_limit_deg) 모두 지원
    if "baseline_deg" in data:
        baseline    = float(data["baseline_deg"])
        dang_mult   = getattr(args, 'dang_mult', AirBalanceMonitor.DANG_MULT)
        warning_deg = baseline * dang_mult
        danger_deg  = warning_deg
    else:
        # 구 형식: theta_limit_deg → baseline 으로 처리
        baseline    = float(data.get("theta_limit_deg", data.get("theta_limit", args.theta_limit)))
        dang_mult   = getattr(args, 'dang_mult', AirBalanceMonitor.DANG_MULT)
        warning_deg = baseline * dang_mult
        danger_deg  = warning_deg

    omega_limit = float(data.get("omega_limit_deg_s", data.get("omega_limit",
                                                                _derive_omega_limit(baseline))))
    profile = SafetyProfile(
        initialized=True,
        baseline_deg=baseline,
        warning_deg=warning_deg,
        danger_deg=danger_deg,
        omega_limit_deg_s=omega_limit,
        source=str(args.profile),
        leg_length_px=float(data.get("leg_length_px", 0.0)),
    )

    if args.webcam:
        src: Union[int, str] = args.camera_index
        live_input = True
    else:
        vp = args.input or find_default_video()
        src = str(vp) if vp else args.camera_index
        live_input = isinstance(src, int)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_clock = FrameClock(live_input=live_input, fps=fps)
    monitor = AirBalanceMonitor(profile=profile)
    estimator   = PoseEstimator(args.model)
    lm_smoother = LandmarkSmoother(tau=0.04)
    leg_filter  = LegMotionFilter()

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        w_px, h_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_px, h_px))

    csv_file = csv_writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "timestamp_s", "timestamp_ms",
            "side", "angle_mode",
            "theta_deg", "omega_deg_s",
            "score", "average_score", "state", "spread_deg",
            "n_real_legs",
            "l_hip_conf", "l_knee_conf", "l_ankle_conf",
            "r_hip_conf", "r_knee_conf", "r_ankle_conf",
        ])

    print(f"[독립 모드] 프로필: {profile.source}")
    print(f"  기준각: {profile.baseline_deg:.1f}°  경고: {profile.warning_deg:.1f}°  "
          f"위험: {profile.danger_deg:.1f}°  각속도: {profile.omega_limit_deg_s:.1f} deg/s")

    spread_limit = profile.warning_deg * 2.0
    last_readings:    List[LegReading]      = []
    last_measurement: Optional[Measurement] = None
    _cnt_monitoring = 0
    _cnt_both_real  = 0
    _cnt_one_extrap = 0
    _cnt_fail       = 0
    _conf_keys = ["l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
    _conf_data: Dict[str, List[float]] = {k: [] for k in _conf_keys}
    frame_index = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_clock.next_seconds(cap, frame_index)
            ts_ms = int(round(ts * 1000.0))
            raw_lm = estimator.detect(frame, ts_ms)
            landmarks = lm_smoother.smooth(raw_lm, ts) if raw_lm is not None else None
            if raw_lm is not None:
                _fc = (
                    _lm_confidence(raw_lm[23]), _lm_confidence(raw_lm[25]), _lm_confidence(raw_lm[27]),
                    _lm_confidence(raw_lm[24]), _lm_confidence(raw_lm[26]), _lm_confidence(raw_lm[28]),
                )
            else:
                _fc = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for _ck, _cv in zip(_conf_keys, _fc):
                _conf_data[_ck].append(_cv)
            h_fr, w_fr = frame.shape[:2]
            raw_readings, spread_angle = extract_leg_readings(
                landmarks, frame,
                args.side, args.visibility_threshold,
                force_2pt=(args.angle_mode == "2pt"),
            )
            readings = leg_filter.process(raw_readings, h_fr, w_fr)
            real = [r for r in readings if not r.is_extrapolated]
            _cnt_monitoring += 1
            n_real = len(real)
            if n_real >= 2:
                _cnt_both_real += 1
            elif n_real == 1:
                _cnt_one_extrap += 1
            else:
                _cnt_fail += 1
            if len(real) == 2:
                spread_angle = AirBalanceMonitor._inter_leg_angle(real[0], real[1])
            measurement = monitor.process_readings(real if real else readings, ts)
            # 다리 벌어짐 초과 시 최소 위험 상태로 강제 전환
            if (spread_angle is not None
                    and spread_angle > spread_limit
                    and measurement is not None
                    and STATE_ORDER[measurement.state] < STATE_ORDER["위험"]):
                measurement.state = "위험"
                measurement.color = STATE_COLORS["위험"]
            if measurement is not None:
                trigger_alert(measurement.state)
                if csv_writer is not None:
                    used = next((r for r in readings if r.side == measurement.side), None)
                    mode = "3pt" if (used and used.knee) else "2pt"
                    csv_writer.writerow([
                        frame_index, f"{ts:.4f}", ts_ms,
                        measurement.side, mode,
                        f"{measurement.theta_deg:.3f}",
                        f"{measurement.omega_deg_s:.3f}",
                        f"{measurement.score:.3f}",
                        f"{measurement.average_score:.3f}",
                        measurement.state,
                        f"{spread_angle:.1f}" if spread_angle is not None else "",
                        n_real,
                        *[f"{v:.3f}" for v in _fc],
                    ])
            tracking_ok = bool(readings)
            disp_readings    = readings    if readings    else last_readings
            disp_measurement = measurement if measurement is not None else last_measurement
            if readings:               last_readings    = readings
            if measurement is not None: last_measurement = measurement
            draw_monitoring_overlay(
                frame, disp_readings, disp_measurement, profile,
                spread_angle=spread_angle, spread_limit=spread_limit,
                tracking_ok=tracking_ok,
            )
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                delay = 1 if live_input else max(1, int(round(1000.0 / fps)))
                cv2.imshow("AirBalance Monitor (standalone)", frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        estimator.close()
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    print(f"[독립 모드] 완료: {frame_index} 프레임 처리")

    if _cnt_monitoring > 0:
        t = _cnt_monitoring
        print("\n┌───────────────────────────────┬────────┬─────────┐")
        print("│           가려짐 보완 성능 통계                   │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print("│ 구분                          │ 프레임 │  비율   │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print(f"│ 양측 실측 감지                │ {_cnt_both_real:6d} │ {_cnt_both_real/t*100:6.1f}% │")
        print(f"│ 단측 외삽 보완                │ {_cnt_one_extrap:6d} │ {_cnt_one_extrap/t*100:6.1f}% │")
        print(f"│ 감지 완전 실패 (양측 누락)    │ {_cnt_fail:6d} │ {_cnt_fail/t*100:6.1f}% │")
        print("├───────────────────────────────┼────────┼─────────┤")
        print(f"│ 모니터링 총 프레임            │ {t:6d} │  100.0% │")
        print("└───────────────────────────────┴────────┴─────────┘")

    if _conf_data["l_hip"]:
        labels = ["L Hip  ", "L Knee ", "L Ankle", "R Hip  ", "R Knee ", "R Ankle"]
        print("\n┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┐")
        print("│              랜드마크 confidence 통계                          │")
        print("├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤")
        print("│ 관절     │  mean  │  min   │  p25   │  p50   │  p75   │  max   │")
        print("├──────────┼────────┼────────┼────────┼────────┼────────┼────────┤")
        for label, key in zip(labels, _conf_keys):
            arr = np.array(_conf_data[key])
            print(f"│ {label} │ {arr.mean():.3f}  │ {arr.min():.3f}  │ "
                  f"{np.percentile(arr,25):.3f}  │ {np.percentile(arr,50):.3f}  │ "
                  f"{np.percentile(arr,75):.3f}  │ {arr.max():.3f}  │")
        print("└──────────┴────────┴────────┴────────┴────────┴────────┴────────┘")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "AirBalance  –  통합 파이프라인\n"
            "  기본: 초기화(YOLO·팀원 코드) → 모니터링(MediaPipe·내 코드)\n"
            "  --profile 지정 시: 초기화 건너뛰고 모니터링만 실행"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  type=Path, help="입력 영상 경로. 미지정 시 같은 폴더 영상 자동 탐색")
    p.add_argument("--webcam", action="store_true", help="웹캠 입력 사용")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--profile", type=Path,
                   help="초기화 완료 JSON 파일 (지정 시 초기화 단계 건너뜀)")
    p.add_argument("--theta-limit",  type=float, default=45.0,
                   help="개인 기준각 기본값 (deg). profile/InitManager 미사용 시 baseline 으로 적용")
    p.add_argument("--omega-limit",  type=float, default=0.0,
                   help="각속도 한계 기본값 (deg/s). 0이면 theta-limit × 3 자동 계산")
    p.add_argument("--dang-mult",    type=float, default=AirBalanceMonitor.DANG_MULT,
                   help=f"위험 임계각 = baseline × dang-mult (기본 {AirBalanceMonitor.DANG_MULT})")
    p.add_argument("--angle-weight", type=float, default=0.85,
                   help="(하위호환용, 미사용)")
    p.add_argument("--persistence",  type=float, default=0.5,
                   help="(하위호환용, 미사용)")
    p.add_argument("--side", choices=("left", "right", "both"), default="both",
                   help="측정할 다리 측 (측면 영상은 left/right 단일 지정 권장)")
    p.add_argument("--visibility-threshold", type=float, default=0.15,
                   help="MediaPipe 랜드마크 신뢰도 최솟값 (기본 0.15; 하늘걷기 기구에서 ankle vis≈0.15~0.30)")
    p.add_argument("--angle-mode", choices=("2pt", "3pt"), default="2pt",
                   help="2pt=측면용 hip-ankle 기울기(기본), 3pt=정면용 hip-knee-ankle 내각")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                   help="MediaPipe PoseLandmarker 모델 경로")
    p.add_argument("--output", type=Path, help="오버레이 결과 영상 저장 경로")
    p.add_argument("--csv",    type=Path, help="프레임별 수치 CSV 저장 경로")
    p.add_argument("--no-display", action="store_true", help="화면 창 없이 처리")
    p.add_argument("--create-profile", action="store_true",
                   help="초기화 완료 후 profile_saved.json 저장만 하고 종료. "
                        "이후 --profile profile_saved.json 으로 다른 영상에 재사용")
    p.add_argument("--loop",       action="store_true",
                   help="영상 끝에서 처음으로 돌아가 반복 (q/ESC로 종료)")
    p.add_argument("--debug-lm",  action="store_true",
                   help="모니터링 구간에서 MediaPipe 33개 랜드마크 전체 표시 (진단용)")
    p.add_argument("--benchmark", action="store_true",
                   help="FPS·지연시간 측정 모드. 화면에 실시간 표시 + 종료 시 통계 출력")
    # ── iVCam (아이폰 카메라 연동) ────────────────────────────────────────────
    p.add_argument("--ivcam", action="store_true",
                   help="iVCam v7 으로 연결된 아이폰 카메라를 자동 탐색하여 입력으로 사용")
    p.add_argument("--list-cameras", action="store_true",
                   help="사용 가능한 카메라 장치 목록을 출력하고 종료 (iVCam 인덱스 확인용)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 카메라 목록 출력 후 종료 ──────────────────────────────────────────────
    if args.list_cameras:
        cameras = scan_cameras()
        if not cameras:
            print("[카메라] 사용 가능한 장치가 없습니다.")
        else:
            print("[카메라 목록]")
            for idx, res in cameras:
                label = " ← 내장 카메라 (MacBook FaceTime)" if idx == 0 else ""
                print(f"  [{idx}]  해상도: {res}{label}")
            print("\niVCam 사용 시: python airbalance.py --ivcam --webcam")
        return

    # ── iVCam 자동 탐색 ──────────────────────────────────────────────────────
    if args.ivcam:
        print("[iVCam] 아이폰 카메라 장치를 탐색합니다...")
        idx = find_iphone_camera_index()
        if idx is None:
            print("[iVCam] 장치를 찾을 수 없습니다.\n"
                  "  1. 아이폰에서 iVCam 앱을 실행하세요.\n"
                  "  2. 맥북과 같은 Wi-Fi 또는 USB로 연결하세요.\n"
                  "  3. 맥에서 iVCam 소프트웨어(v7.x)가 실행 중인지 확인하세요.\n"
                  "  4. --list-cameras 로 인덱스를 직접 확인 후 --camera-index N 으로 지정할 수 있습니다.")
            return
        args.camera_index = idx
        args.webcam = True
        print(f"[iVCam] 인덱스 [{idx}] 장치로 연결합니다. (아이폰 카메라)")

    # omega_limit 자동 계산
    if args.omega_limit <= 0:
        args.omega_limit = _derive_omega_limit(args.theta_limit)

    if args.profile is not None:
        run_standalone(args)
    else:
        run_integrated(args)


if __name__ == "__main__":
    main()