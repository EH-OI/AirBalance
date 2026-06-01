# visualizer.py
# 포즈 분석 시각화 피드백을 제공하는 클래스

import math
import cv2
from constants import (
    InitState, CONF_THRESHOLD,
    KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_KNEE,
    KP_RIGHT_KNEE, KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    COLOR_HIP_ANKLE, COLOR_KNEE, COLOR_SKELETON,
    COLOR_LEG_LINE, COLOR_INIT_TEXT, COLOR_DONE_TEXT,
    COLOR_BBOX, COLOR_BAR_BG, COLOR_BAR_FG
)

class Visualizer:
    def __init__(self):
        # Visualizer 초기화
        pass

    def draw_skeleton(self, frame, keypoints):
        # 주요 하체 키포인트와 연결선(스켈레톤)을 프레임 상에 그립니다.
        # 파라미터 frame: 이미지 프레임 (numpy array)
        # 파라미터 keypoints: 17개 키포인트 리스트 [(x, y, conf), ...]
        
        kpts = {}
        target_indices = [
            KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_KNEE, 
            KP_RIGHT_KNEE, KP_LEFT_ANKLE, KP_RIGHT_ANKLE
        ]
        
        # 신뢰도 임계값 이상인 키포인트의 정수 좌표만 수집하여 딕셔너리에 매핑
        for idx in target_indices:
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf >= CONF_THRESHOLD:
                    kpts[idx] = (int(x), int(y))

        # [관절 점 그리기]
        for idx, pt in kpts.items():
            if idx in [KP_LEFT_HIP, KP_RIGHT_HIP, KP_LEFT_ANKLE, KP_RIGHT_ANKLE]:
                # 엉덩이와 발목 관절: 지정된 노란색(COLOR_HIP_ANKLE), 반지름 6, 안쪽을 꽉 채워서(-1) 그림
                cv2.circle(frame, pt, 6, COLOR_HIP_ANKLE, -1)
            elif idx in [KP_LEFT_KNEE, KP_RIGHT_KNEE]:
                # 무릎 관절: 지정된 초록색(COLOR_KNEE), 반지름 6, 안쪽을 꽉 채워서(-1) 그림
                cv2.circle(frame, pt, 6, COLOR_KNEE, -1)

        # [관절 연결선 그리기 헬퍼 함수]
        def draw_line(p1_idx, p2_idx):
            # 두 키포인트가 모두 감지 기준 신뢰도를 통과한 경우에만 선을 그림
            if p1_idx in kpts and p2_idx in kpts:
                cv2.line(frame, kpts[p1_idx], kpts[p2_idx], COLOR_SKELETON, 2)

        # 왼쪽 다리: 엉덩이 -> 무릎 -> 발목 연결선 그리기
        draw_line(KP_LEFT_HIP, KP_LEFT_KNEE)
        draw_line(KP_LEFT_KNEE, KP_LEFT_ANKLE)
        
        # 오른쪽 다리: 엉덩이 -> 무릎 -> 발목 연결선 그리기
        draw_line(KP_RIGHT_HIP, KP_RIGHT_KNEE)
        draw_line(KP_RIGHT_KNEE, KP_RIGHT_ANKLE)
        
        # 골반선: 왼쪽 엉덩이 -> 오른쪽 엉덩이 연결선 그리기
        draw_line(KP_LEFT_HIP, KP_RIGHT_HIP)

    @staticmethod
    def _dashed_line(frame, pt1, pt2, color, thickness, dash=18):
        """점선을 pt1→pt2 방향으로 그립니다 (OpenCV 기본 지원 없음)."""
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            return
        nx, ny = dx / dist, dy / dist
        t = 0.0
        draw = True
        while t < dist:
            t_end = min(t + dash, dist)
            if draw:
                s = (int(pt1[0] + nx * t),     int(pt1[1] + ny * t))
                e = (int(pt1[0] + nx * t_end),  int(pt1[1] + ny * t_end))
                cv2.line(frame, s, e, color, thickness)
            t += dash
            draw = not draw

    def draw_leg_length_line(self, frame, keypoints, l_avg, bbox=None, display_slots=None):
        # 초기화 진행(IN_PROGRESS) 중 다리 벡터를 그립니다.
        #
        # display_slots 가 제공되면 (init_manager가 처리한 NMS·외삽 결과) 그것을 직접 그립니다.
        # - 실측 벡터: 노란색 실선, "#1: 45.1deg" 레이블
        # - 외삽 벡터: 회색 점선, "#N(est)" 레이블 (가려진 다리 위치 추정)
        # display_slots 없으면 기존 방식(keypoints에서 직접 계산)으로 폴백합니다.
        # 파라미터 l_avg: 현재까지 수집된 최대 각도 (deg)

        h, w = frame.shape[:2]

        if display_slots:
            for i, slot in enumerate(display_slots):
                hip    = (int(slot['hip'][0]),    int(slot['hip'][1]))
                ankle  = (int(slot['ankle'][0]),  int(slot['ankle'][1]))
                is_ext = slot.get('extrap', False)

                if is_ext:
                    color = (160, 160, 160)   # 회색: 외삽(추정) 벡터
                    self._dashed_line(frame, hip, ankle, color, 2)
                    label = f"#{i+1}(est)"
                else:
                    color = COLOR_LEG_LINE
                    cv2.line(frame, hip, ankle, color, 4)
                    label = f"#{i+1}: {slot['angle']:.1f}deg"

                cv2.circle(frame, hip,   6, COLOR_HIP_ANKLE if not is_ext else color, -1)
                cv2.circle(frame, ankle, 6, COLOR_HIP_ANKLE if not is_ext else color, -1)
                mx = (hip[0] + ankle[0]) // 2
                my = (hip[1] + ankle[1]) // 2
                cv2.putText(frame, label, (mx + 10, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIP_ANKLE, 2)
        else:
            # ── 폴백: keypoints에서 직접 계산 (display_slots 없을 때) ──────────
            bbox_h = float(bbox[3] - bbox[1]) if bbox is not None else h * 0.6
            min_len = bbox_h * 0.40
            candidates = []
            for hip_idx, ankle_idx in [
                (KP_LEFT_HIP, KP_LEFT_ANKLE),
                (KP_RIGHT_HIP, KP_RIGHT_ANKLE),
            ]:
                if hip_idx >= len(keypoints) or ankle_idx >= len(keypoints):
                    continue
                hx, hy, hc = keypoints[hip_idx]
                ax, ay, ac = keypoints[ankle_idx]
                if hc < CONF_THRESHOLD or ac < CONF_THRESHOLD:
                    continue
                length = math.sqrt((ax - hx) ** 2 + (ay - hy) ** 2)
                if length < min_len:
                    continue
                dy = ay - hy
                angle = math.degrees(math.acos(max(-1.0, min(1.0, dy / length))))
                candidates.append({
                    "hip": (int(hx), int(hy)), "ankle": (int(ax), int(ay)),
                    "length": length, "conf": min(hc, ac), "angle": angle,
                })
            candidates.sort(key=lambda c: -(c["length"] * c["conf"]))
            selected = []
            for cand in candidates:
                if not any(abs(cand["hip"][0] - s["hip"][0]) < w * 0.06 for s in selected):
                    selected.append(cand)
            for i, cand in enumerate(selected):
                cv2.line(frame, cand["hip"], cand["ankle"], COLOR_LEG_LINE, 4)
                cv2.circle(frame, cand["hip"],   6, COLOR_HIP_ANKLE, -1)
                cv2.circle(frame, cand["ankle"], 6, COLOR_HIP_ANKLE, -1)
                mx = (cand["hip"][0] + cand["ankle"][0]) // 2
                my = (cand["hip"][1] + cand["ankle"][1]) // 2
                cv2.putText(frame, f"#{i+1}: {cand['angle']:.1f}deg",
                            (mx + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIP_ANKLE, 2)

        # 누적 최대 각도 표시
        if l_avg > 0:
            cv2.putText(frame, f"Max: {l_avg:.1f} deg",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INIT_TEXT, 2)

    def draw_progress_bar(self, frame, progress):
        # 화면 하단에 초기화 진행률 표시 바를 그립니다.
        # 파라미터 frame: 이미지 프레임 (numpy array)
        # 파라미터 progress: 진행률 (0.0 ~ 1.0 실수)
        
        h, w = frame.shape[:2]  # 프레임의 세로 및 가로 크기 획득
        
        # 1. 배경바 그리기 (좌우 30픽셀 여백 유지, 높이는 하단 50px ~ 30px 영역)
        cv2.rectangle(frame, (30, h - 50), (w - 30, h - 30), COLOR_BAR_BG, -1)
        
        # 2. 진행 상태에 따른 전경바 그리기 (진행 비율에 따라 전경바의 우측 x좌표 연산)
        bar_x2 = int(30 + (w - 60) * progress)
        cv2.rectangle(frame, (30, h - 50), (bar_x2, h - 30), COLOR_BAR_FG, -1)

    def draw_init_overlay(self, frame, keypoints, progress, state_name, l_avg,
                          bbox=None, display_slots=None, debug_info=None):
        # 초기화 진행 중 상태의 오버레이를 그립니다.

        self.draw_skeleton(frame, keypoints)
        self.draw_leg_length_line(frame, keypoints, l_avg,
                                  bbox=bbox, display_slots=display_slots)
        self.draw_progress_bar(frame, progress)
        cv2.putText(frame, f"[{state_name}] Initializing... {int(progress*100)}%",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_INIT_TEXT, 2)

        # 디버그 텍스트 (NMS·방향 필터 수치 실시간 표시)
        if debug_info:
            d = debug_info
            line1 = (f"[DEBUG] AnkleDist: {d.get('ankle_dist', 0):.1f}px"
                     f"  /  NMS_Thresh: {d.get('nms_thresh', 0):.1f}px")
            line2 = (f"[DEBUG] DirAngle: {d.get('angle_diff', 0):.1f}deg"
                     f"  /  Dir_Thresh: {d.get('dir_thresh', 0):.0f}deg"
                     f"  /  NMS_Passed: {d.get('nms_passed', 0)}")
            cv2.putText(frame, line1, (10, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
            cv2.putText(frame, line2, (10, frame.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

    def draw_done_overlay(self, frame, keypoints, threshold, detection):
        # 초기화 완료 상태의 오버레이(스켈레톤, 완료 텍스트, 바운딩 박스)를 그립니다.
        # 파라미터 frame: 이미지 프레임 (numpy array)
        # 파라미터 keypoints: 17개 키포인트 리스트 [(x, y, conf), ...]
        # 파라미터 threshold: 계산 완료된 안전 임계각
        # 파라미터 detection: YOLOProcessor.detect() 반환 데이터 딕셔너리
        
        self.draw_skeleton(frame, keypoints)  # 1. 하체 뼈대 및 관절 렌더링
        
        # 2. 상단 좌측에 준비 완료 및 산정된 안전 임계각 표시
        cv2.putText(frame, f"Ready | Threshold: {threshold:.1f} deg",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DONE_TEXT, 2)
        
        # 3. 감지된 사용자 바운딩박스 영역 사각형 그리기 (주황색, 두께 2)
        if detection and "bbox" in detection:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_BBOX, 2)
