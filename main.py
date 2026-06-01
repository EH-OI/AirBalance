# main.py
# 비디오 스트림 처리, 포즈 검출 및 시각화 메인 파이프라인 스크립트

import time
import cv2
from constants import (
    InitState, COLOR_INIT_TEXT, COLOR_DONE_TEXT, INIT_DURATION
)
from video_source import VideoSourceManager
from yolo_processor import YOLOProcessor
from init_manager import InitializationManager
from visualizer import Visualizer

def run_pipeline(source):
    # 비디오 소스를 입력받아 YOLO 포즈 추정 및 시각화 오버레이 파이프라인을 실행합니다.
    # 파라미터 source: int(웹캠 인덱스) 또는 str(로컬 비디오 파일 경로 또는 유튜브 URL)
    
    # 1. 입력 비디오 제어기 초기화 및 로드
    video = VideoSourceManager(source)
    video.open()

    # 2. YOLO 인공지능 포즈 감지기 객체 생성 및 가중치 파일 로드
    yolo = YOLOProcessor()
    yolo.load_model()

    # 3. 사용자 초기화 및 상태 관리 인스턴스 생성
    init_manager = InitializationManager()

    # 4. 실시간 피드백 렌더러(비주얼라이저) 객체 생성
    visualizer = Visualizer()

    try:
        # 실시간 비디오 프레임 루프 가동
        while True:
            # 1프레임 캡처 시도
            ret, frame = video.read_frame()
            if not ret:
                print("비디오 스트림이 종료되었거나 프레임을 읽을 수 없습니다.")
                break

            # YOLO 모델로 사용자의 관절 뼈대 및 바운딩 박스 추론 진행
            detection = yolo.detect(frame)
            
            # 사용자 존재 및 측정 데이터 누적 연산 진행 (매 프레임 호출 필수)
            init_manager.process_frame(detection)

            # [분기 1] 초기화(7초 신체 학습) 단계가 아직 완료되지 않은 상태
            if not init_manager.is_done():
                # 사용자가 카메라에 잡혀 측정(IN_PROGRESS) 중인 상태
                if detection is not None and init_manager.state == InitState.IN_PROGRESS:
                    # 7초 동안 흐른 시간 비율 계산
                    elapsed = time.time() - init_manager.init_start_time
                    progress = min(1.0, elapsed / INIT_DURATION)
                    
                    # 가장 최근 프레임에서 계산된 실시간 평균 다리 길이 도출 (없으면 0.0)
                    l_avg = init_manager.landmark_buffer[-1] if init_manager.landmark_buffer else 0.0
                    
                    # 초기화용 오버레이 화면(관절선 + 다리 길이 가이드선 + 진행 게이지바 + 텍스트) 그리기
                    visualizer.draw_init_overlay(
                        frame, 
                        detection["keypoints"], 
                        progress, 
                        init_manager.get_state(), 
                        l_avg
                    )
                # 사용자가 감지되지 않았거나 완전히 초기 대기 상태(NONE)인 상태
                else:
                    # 화면 좌상단에 초록색 대기 텍스트 표시
                    cv2.putText(
                        frame, 
                        "Waiting for person...", 
                        (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        COLOR_INIT_TEXT, 
                        2
                    )
            # [분기 2] 초기화(7초 학습)가 완료되어 실시간 모니터링이 준비(DONE)된 상태
            else:
                # 초기화 단계에서 도출된 사용자 고유 안전 임계각 획득
                threshold = init_manager.get_threshold()
                
                # 사용자가 계속 실시간 감지되고 있는 경우
                if detection is not None:
                    # 완료 오버레이(뼈대 스켈레톤 + 주황색 타겟 바운딩박스 + 임계각 결과 텍스트) 그리기
                    visualizer.draw_done_overlay(
                        frame, 
                        detection["keypoints"], 
                        threshold, 
                        detection
                    )
                # 사용자가 일시적으로 가려졌거나 미감지된 경우 (3초 이상 지속 시 NONE 리셋됨)
                else:
                    # 대기 정보 및 안전 임계각 텍스트만 주황색으로 상단에 표시
                    cv2.putText(
                        frame, 
                        f"Ready | Threshold: {threshold:.1f} deg", 
                        (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        COLOR_DONE_TEXT, 
                        2
                    )

            # 피드백 오버레이가 합성 완료된 최종 영상 프레임을 윈도우 창으로 출력
            cv2.imshow("AirBalance Pose Analyzer", frame)
            
            # ESC 키(ASCII 27) 또는 키보드 'q' 키가 입력되면 루프를 즉시 빠져나와 안전 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
    finally:
        # 비디오 장치 사용 해제 및 생성된 OpenCV 비디오 모니터링 창을 안전하게 모두 닫음
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 기본 옵션: 0번 웹캠 장치 연결
    # 유튜브 스트림(예: "https://www.youtube.com/watch?v=aclHkVaku9U") 또는 비디오 파일명을 인자로 변경할 수 있습니다.
    run_pipeline(0)
