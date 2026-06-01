# init_manager.py
# 포즈 분석 시스템의 사용자 초기화 관리 클래스

import time
import math
from constants import (
    InitState, INIT_DURATION, ABSENCE_THRESHOLD, ALPHA,
    CONF_THRESHOLD, KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE, LEG_LENGTH_TOLERANCE,
    MISMATCH_THRESHOLD
)

class InitializationManager:
    _SPATIAL_NMS_FRAC: float = 0.20  # 발목 간 최소 분리 거리 (bbox_h 비율)
    _DIR_THRESH_DEG:   float = 30.0  # 두 벡터 방향 사잇각 임계값 (deg) — 이하면 같은 다리로 판정
    _MAX_EXTRAP_INIT:  int   = 5     # 최대 연속 외삽 프레임 수 (초기화 중)

    def __init__(self):
        # InitializationManager 초기화
        self.state: InitState = InitState.NONE      # 초기 상태 설정 (초기화 전 대기 상태)
        self.init_start_time: float = 0.0          # 초기화(데이터 수집)가 시작된 절대 시간 저장용
        self.absence_start_time: float | None = None # 사용자가 화면에서 보이지 않기 시작한 절대 시간 저장용
        self.landmark_buffer: list[float] = []       # 초기화 진행 중 프레임별 최대 다리 각도(deg)를 저장하는 버퍼
        self.threshold_angle: float | None = None    # 사용자의 신체 조건에 따른 안전 임계각(최종 결과값)

        # 신규 사용자 전환 감지를 위한 변수 추가
        self.leg_length_px: float | None = None      # 초기화 완료된 다리 길이 저장용 (사용자 변경 감지용)
        self._leg_lengths: list[float] = []          # 다리 길이 참조값 계산용 내부 버퍼
        self.current_track_id: int | None = None     # 현재 추적 중인 사용자의 트래킹 ID
        self.mismatch_start_time: float | None = None # 트래킹 ID 혹은 다리 길이 불일치가 시작된 시간

        # 공간 분리(Spatial NMS) + 외삽용 슬롯 상태
        self._init_slots: list = [None, None]        # 슬롯별 마지막 유효 벡터 (연속성 추적)
        self._current_display_slots: list = []       # 현재 프레임 시각화 슬롯 (외삽 포함)
        self._debug_info: dict = {}                  # 디버그 텍스트용 최신 수치

    def _reset_state(self):
        # 모든 데이터 및 상태를 초기 NONE 상태로 완전 리셋합니다.
        self.state = InitState.NONE
        self.landmark_buffer = []
        self._leg_lengths = []
        self.threshold_angle = None
        self.leg_length_px = None
        self.current_track_id = None
        self.absence_start_time = None
        self.mismatch_start_time = None
        self._init_slots = [None, None]
        self._current_display_slots = []
        self._debug_info = {}

    # ------------------------------------------------------------------
    # 내부: Spatial NMS + 슬롯 매칭 + 외삽
    # ------------------------------------------------------------------

    def _process_candidates(self, candidates: list, bbox_h: float) -> tuple:
        """
        3단계 파이프라인으로 후보 벡터를 정제한다.

        1. 신뢰도×길이 내림차순 정렬 + atan2 방향각 계산
        2. atan2 방향 기반 NMS (L/R 레이블 완전 무시):
             #1 확정 후, 다음 후보와의 방향 사잇각 < _DIR_THRESH_DEG(30°) 이면
             같은 다리로 판정해 폐기. ≥ 30° 이면 다른 다리로 인정해 #2로 채택.
        3. 슬롯 매칭 + 외삽 (기존과 동일)

        반환:
          real_angles : 실제 감지된(비외삽) 각도 목록 → landmark_buffer 갱신에 사용
          all_slots   : 시각화용 슬롯 목록 (extrap=True 면 외삽, False 면 실측)
        """
        # 1. atan2 방향각 추가 후 신뢰도×길이 내림차순 정렬
        for cand in candidates:
            dx = cand['ankle'][0] - cand['hip'][0]
            dy = cand['ankle'][1] - cand['hip'][1]
            cand['dir_deg'] = math.degrees(math.atan2(dy, dx))

        sorted_c = sorted(candidates, key=lambda c: -(c['conf'] * c['length']))

        # 2. 방향 기반 NMS: 사잇각 기준으로 중복 다리 제거
        nms: list = []
        ankle_dist = 0.0
        angle_diff = 0.0
        for cand in sorted_c:
            if not nms:
                nms.append(cand)  # #1은 무조건 확정
            else:
                diff = abs(cand['dir_deg'] - nms[0]['dir_deg'])
                if diff > 180.0:
                    diff = 360.0 - diff
                ankle_dist = math.sqrt(
                    (cand['ankle'][0] - nms[0]['ankle'][0]) ** 2 +
                    (cand['ankle'][1] - nms[0]['ankle'][1]) ** 2
                )
                angle_diff = diff
                if diff >= self._DIR_THRESH_DEG:  # 30° 이상 → 다른 다리
                    nms.append(cand)
                    break  # 다리는 최대 2개

        self._debug_info = {
            'ankle_dist': ankle_dist,
            'nms_thresh': bbox_h * self._SPATIAL_NMS_FRAC,
            'angle_diff': angle_diff,
            'dir_thresh': self._DIR_THRESH_DEG,
            'nms_passed': len(nms),
        }

        # 2. 슬롯 매칭: 이전 슬롯과의 발목 근접도로 greedy 배정
        new_slots: list = [None, None]
        used: set = set()

        for si in range(2):
            prev = self._init_slots[si]
            if prev is None:
                continue
            best_vi, best_d = -1, float('inf')
            for vi, cand in enumerate(nms):
                if vi in used:
                    continue
                d = math.sqrt((cand['ankle'][0] - prev['ankle'][0]) ** 2 +
                              (cand['ankle'][1] - prev['ankle'][1]) ** 2)
                if d < best_d:
                    best_d, best_vi = d, vi
            if best_vi >= 0:
                new_slots[si] = {**nms[best_vi], 'extrap': False, 'extrap_count': 0}
                used.add(best_vi)
            elif prev.get('extrap_count', 0) < self._MAX_EXTRAP_INIT:
                # 정적 외삽: 이전 위치 유지
                new_slots[si] = {**prev, 'extrap': True,
                                 'extrap_count': prev.get('extrap_count', 0) + 1}

        # 3. 미할당 NMS 벡터를 빈 슬롯에 순서대로 배치
        for vi, cand in enumerate(nms):
            if vi not in used:
                for si in range(2):
                    if new_slots[si] is None:
                        new_slots[si] = {**cand, 'extrap': False, 'extrap_count': 0}
                        used.add(vi)
                        break

        self._init_slots = new_slots
        real_angles = [s['angle'] for s in new_slots if s and not s.get('extrap')]
        all_slots   = [s for s in new_slots if s is not None]
        return real_angles, all_slots

    def get_display_slots(self) -> list:
        """현재 프레임의 시각화용 슬롯 목록 반환 (외삽 포함)."""
        return [s for s in self._current_display_slots if s]

    def get_debug_info(self) -> dict:
        """NMS·방향 필터 디버그 수치 반환."""
        return self._debug_info

    def process_frame(self, detection: dict | None) -> dict | None:
        # 매 프레임 감지 결과를 처리하여 초기화 과정을 갱신하고 맞춤형 안전 임계각을 계산합니다.
        # 파라미터 detection: YOLOProcessor.detect()의 반환값 (감지 실패 시 None)
        # 반환값: 초기화 완료 시 {"status": "DONE", "threshold_angle": float, "leg_length_px": float} 반환,
        #         그 외 상태 변경 시 None 반환
        
        current_time = time.time()  # 현재 시스템 절대 시간 획득

        # ==========================================
        # [DONE 상태] 이탈 감지 및 자동 리셋 제어
        # ==========================================
        if self.state == InitState.DONE:
            # 사용자가 감지되지 않은 상황
            if detection is None:
                # 미감지 카운트다운 타이머가 시작되지 않은 경우 현재 시간을 기록
                if self.absence_start_time is None:
                    self.absence_start_time = current_time
                # 미감지 상태가 지속되어 임계 임계값(ABSENCE_THRESHOLD = 3초)을 넘은 경우
                elif current_time - self.absence_start_time >= ABSENCE_THRESHOLD:
                    # 다른 사용자로의 교체 혹은 운동 종료로 판정하여 데이터 및 상태를 리셋
                    self._reset_state()
            else:
                # 사용자가 계속 감지되고 있으면 미감지 타이머 리셋
                self.absence_start_time = None
                
                # 다른 사용자 진입 여부를 검증 (하이브리드: 트래킹 ID 및 다리 길이 비교)
                track_id = detection.get("track_id")
                
                # 1. 트래킹 ID 불일치 확인
                is_id_mismatch = (track_id is not None and 
                                  self.current_track_id is not None and 
                                  track_id != self.current_track_id)
                
                # 2. 실시간 다리 길이 계산 및 기존 다리 길이와 비교
                is_leg_mismatch = False
                if self.leg_length_px is not None:
                    keypoints = detection["keypoints"]
                    l_hip = keypoints[KP_LEFT_HIP]
                    r_hip = keypoints[KP_RIGHT_HIP]
                    l_ankle = keypoints[KP_LEFT_ANKLE]
                    r_ankle = keypoints[KP_RIGHT_ANKLE]
                    
                    if (l_hip[2] >= CONF_THRESHOLD and
                        r_hip[2] >= CONF_THRESHOLD and
                        l_ankle[2] >= CONF_THRESHOLD and
                        r_ankle[2] >= CONF_THRESHOLD):
                        
                        L_left = math.sqrt((l_ankle[0] - l_hip[0])**2 + (l_ankle[1] - l_hip[1])**2)
                        L_right = math.sqrt((r_ankle[0] - r_hip[0])**2 + (r_ankle[1] - r_hip[1])**2)
                        L_avg = (L_left + L_right) / 2.0
                        
                        # 오차 비율 계산
                        err_ratio = abs(L_avg - self.leg_length_px) / self.leg_length_px
                        if err_ratio > LEG_LENGTH_TOLERANCE:
                            is_leg_mismatch = True

                # 불일치 조건 성립 시
                if is_id_mismatch or is_leg_mismatch:
                    if self.mismatch_start_time is None:
                        self.mismatch_start_time = current_time
                    elif current_time - self.mismatch_start_time >= MISMATCH_THRESHOLD:
                        # 1.5초 이상 불일치가 지속되면 다른 사용자 진입으로 판정하여 리셋
                        self._reset_state()
                else:
                    # 일시적 불일치였던 경우 타이머 초기화
                    self.mismatch_start_time = None
                    
                    # 만약 트래킹 ID가 유실되었다가 다시 안정적으로 획득된 경우 업데이트
                    if track_id is not None and self.current_track_id is None:
                        self.current_track_id = track_id
            return None

        # ==========================================
        # [NONE 상태] 사용자 신규 진입 시 초기화 절차 시작
        # ==========================================
        if self.state == InitState.NONE:
            # 사용자가 감지되지 않으면 시작하지 않고 계속 대기
            if detection is None:
                return None
            # 사용자가 처음으로 정상 감지되면 초기화 진행 중(IN_PROGRESS) 상태로 전환
            self.state = InitState.IN_PROGRESS
            self.init_start_time = current_time  # 초기화 시작 절대 시간 기록
            self.landmark_buffer = []            # 측정 수치 버퍼 초기화
            self.current_track_id = detection.get("track_id") # 현재 감지된 사용자의 트래킹 ID 저장
            self.mismatch_start_time = None

        # ==========================================
        # [IN_PROGRESS 상태] 7초 동안 신체 데이터 누적 및 안전 각도 계산
        # ==========================================
        if self.state == InitState.IN_PROGRESS:
            # 감지 정보가 프레임에 존재할 경우에만 측정 연산 실행
            if detection is not None:
                # 초기화 도중 갑자기 트래킹 ID가 바뀌면 다른 사람으로 인지하고 타이머 재설정
                track_id = detection.get("track_id")
                if (track_id is not None and 
                    self.current_track_id is not None and 
                    track_id != self.current_track_id):
                    self.landmark_buffer = []
                    self.current_track_id = track_id
                    self.init_start_time = current_time
                    return None

                keypoints = detection["keypoints"]
                bbox = detection.get("bbox")
                bbox_h = float(bbox[3] - bbox[1]) if bbox is not None else 0.0
                min_leg_len = bbox_h * 0.40 if bbox_h > 0 else 0.0

                # L/R 구분 없이 신뢰도·해부학적 필터를 통과한 후보 벡터 수집
                candidates: list = []
                for hip_idx, ankle_idx in [
                    (KP_LEFT_HIP, KP_LEFT_ANKLE),
                    (KP_RIGHT_HIP, KP_RIGHT_ANKLE),
                ]:
                    hx, hy, hc = keypoints[hip_idx]
                    ax, ay, ac = keypoints[ankle_idx]
                    if hc < CONF_THRESHOLD or ac < CONF_THRESHOLD:
                        continue
                    length = math.sqrt((ax - hx) ** 2 + (ay - hy) ** 2)
                    if min_leg_len > 0 and length < min_leg_len:
                        continue
                    dy = ay - hy
                    angle_deg = math.degrees(
                        math.acos(max(-1.0, min(1.0, dy / length)))
                    )
                    candidates.append({
                        'hip': (hx, hy), 'ankle': (ax, ay),
                        'length': length, 'conf': min(hc, ac), 'angle': angle_deg,
                    })

                # Spatial NMS + 슬롯 추적 + 외삽
                real_angles, display_slots = self._process_candidates(candidates, bbox_h)
                self._current_display_slots = display_slots

                # 실측(비외삽) 각도만 버퍼에 누적
                if real_angles:
                    self.landmark_buffer.append(max(real_angles))
                    for s in display_slots:
                        if not s.get('extrap'):
                            self._leg_lengths.append(s['length'])

            else:
                # 감지 없음: 기존 슬롯 외삽만 수행
                _, display_slots = self._process_candidates([], 0.0)
                self._current_display_slots = display_slots

            # 초기화 시작 후 경과 시간 계산
            elapsed_time = current_time - self.init_start_time
            # 아직 데이터 수집 설정 시간(INIT_DURATION = 7초)을 채우지 못한 경우 리턴
            if elapsed_time < INIT_DURATION:
                return None
            # 7초가 지난 경우 데이터 가공 및 완료 처리 진행
            else:
                # 7초 동안 유효한 다리 길이 데이터가 한 번도 누적되지 않은 경우
                if not self.landmark_buffer:
                    print("[알림] 7초 동안 유효한 신체 데이터를 수집하지 못했습니다. 측정을 다시 시작합니다.")
                    self._reset_state()
                    return None
                
                # 상위 15% 프레임 최대각도의 평균을 임계각으로 설정 (이상치에 강건한 Max 추정)
                sorted_angles = sorted(self.landmark_buffer)
                top_count = max(1, int(len(sorted_angles) * 0.15))
                self.threshold_angle = sum(sorted_angles[-top_count:]) / top_count

                # 다리 길이 중앙값: 사용자 변경 감지(DONE 상태 mismatch 검사)용
                if self._leg_lengths:
                    sorted_lens = sorted(self._leg_lengths)
                    L_mean = sorted_lens[len(sorted_lens) // 2]
                else:
                    L_mean = 0.0

                # 상태 완료 정보 저장
                self.leg_length_px = L_mean
                self.state = InitState.DONE
                self.absence_start_time = None
                self.mismatch_start_time = None
                
                # 분석에 필요한 핵심 초기 세팅값을 딕셔너리로 묶어 전달
                return {
                    "status": "DONE",
                    "threshold_angle": self.threshold_angle,
                    "leg_length_px": L_mean
                }
                
        return None

    def is_done(self) -> bool:
        # 초기화가 성공적으로 완료(DONE)되어 서비스 준비가 되었는지 여부를 반환합니다.
        return self.state == InitState.DONE

    def get_threshold(self) -> float | None:
        # 초기화 완료 상태일 때 계산된 안전 임계각을 반환합니다. 초기화 전이면 None을 반환합니다.
        if self.state == InitState.DONE:
            return self.threshold_angle
        return None

    def get_state(self) -> str:
        # 현재 진행 상태(NONE, IN_PROGRESS, DONE)의 문자열 이름을 반환합니다.
        return self.state.value
