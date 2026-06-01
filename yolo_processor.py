# yolo_processor.py
# YOLO11n-pose 모델을 이용하여 영상 프레임 내 사람의 포즈(키포인트)를 감지하는 클래스

import numpy as np
from constants import MIN_BBOX_AREA, CONF_THRESHOLD
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self):
        # YOLO 포즈 프로세서 초기화
        self.model = None  # YOLO 모델 객체를 담을 변수 초기화

    def load_model(self):
        # YOLO11n-pose 사전 학습된 가중치 모델을 메모리에 로드합니다.
        # 로컬 경로에 모델 파일이 없으면 인터넷에서 자동으로 다운로드하여 로드함
        self.model = YOLO("yolo11n-pose.pt")

    def detect(self, frame) -> dict | None:
        # 입력 프레임에서 사람을 감지하고, 유효 면적 이상의 사람 중 가장 바운딩 박스가 큰 사용자의 정보를 반환합니다.
        # 파라미터 frame: 입력 이미지 프레임 (numpy array)
        # 반환값: 감지된 사람의 키포인트, 바운딩 박스 정보 딕셔너리. 감지 실패 시 None 반환.
        # 
        # 반환 딕셔너리 구조:
        # {
        #     "keypoints": [(x1, y1, conf1), (x2, y2, conf2), ...],  # 17개 COCO 키포인트 (x좌표, y좌표, 감지 신뢰도)
        #     "bbox": (x1, y1, x2, y2),  # 감지된 바운딩 박스 좌표 (좌상단 x, y, 우하단 x, y)
        #     "bbox_area": float  # 바운딩 박스의 면적 (픽셀 제곱 단위)
        # }
        
        # 모델 로드 여부 검증
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # YOLO 모델로 이미지 프레임 추론 및 객체 추적(Tracking) 수행
        results = self.model.track(frame, persist=True, verbose=False)
        if not results or len(results) == 0:
            return None  # 검출된 결과가 아예 없으면 None 리턴

        result = results[0]  # 단일 프레임에 대한 검출 결과 객체 획득
        boxes = result.boxes
        keypoints_obj = result.keypoints

        # 바운딩 박스나 키포인트 세부 정보가 존재하지 않는 예외 상황 처리
        if boxes is None or len(boxes) == 0 or keypoints_obj is None:
            return None

        # GPU 메모리에 올라와 있는 Bounding Box 텐서 좌표 데이터를 CPU로 이동 후 Numpy 배열로 변환
        xyxy_list = boxes.xyxy.cpu().numpy()
        
        # 각 바운딩 박스에 대응되는 트래킹 ID 리스트 획득 (트래커 미할당 시 None 일 수 있음)
        track_ids = None
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()
        
        # 키포인트 데이터 좌표 및 신뢰도 정보 추출 ([N, 17, 3] 형태의 텐서 데이터)
        kpts_tensor = keypoints_obj.data
        if kpts_tensor is None or len(kpts_tensor) == 0:
            return None  # 키포인트 텐서가 비어 있으면 None 리턴
            
        kpts_list = kpts_tensor.cpu().numpy()

        max_area = -1.0  # 가장 큰 면적을 추적하기 위한 변수
        best_idx = -1    # 최적의 대상(운동 기구 사용자) 인덱스를 저장할 변수

        # 프레임 안의 모든 감지된 객체 루프 순회
        for idx, box in enumerate(xyxy_list):
            x1, y1, x2, y2 = box
            # 바운딩 박스의 넓이(면적) 계산
            area = float((x2 - x1) * (y2 - y1))
            
            # 1. 설정된 최소 유효 바운딩 박스 면적(MIN_BBOX_AREA)보다 작은 객체는 배경 노이즈로 보고 제외
            if area < MIN_BBOX_AREA:
                continue
                
            # 2. 유효 객체 중 가장 면적이 큰 대상(카메라와 가장 가깝고 명확한 기구 사용자) 선택
            if area > max_area:
                max_area = area
                best_idx = idx

        # 조건(최소 면적 조건 충족 및 객체 존재)을 만족하는 주 대상이 감지되지 않은 경우
        if best_idx == -1:
            return None

        # 선정된 대상의 키포인트(17개 관절 좌표) 정보 정제
        selected_kpts = kpts_list[best_idx]  # Shape: (17, 3) -> 각 행은 (x, y, confidence)
        formatted_keypoints = [tuple(kp) for kp in selected_kpts]  # tuple 형태로 정제하여 리스트 구성
        selected_bbox = tuple(xyxy_list[best_idx])  # 바운딩박스 좌표 튜플 변환
        selected_track_id = int(track_ids[best_idx]) if track_ids is not None else None

        # 정제된 핵심 정보만 딕셔너리로 패킹하여 반환
        return {
            "keypoints": formatted_keypoints,
            "bbox": selected_bbox,
            "bbox_area": max_area,
            "track_id": selected_track_id
        }
