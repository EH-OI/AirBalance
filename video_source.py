# video_source.py
# 비디오 입력 소스(웹캠, 로컬 파일, 유튜브 스트림)를 통합 관리하는 클래스

import cv2
import numpy as np
import yt_dlp
from constants import MIN_BBOX_AREA, CONF_THRESHOLD

class VideoSourceManager:
    def __init__(self, source: str | int):
        # 비디오 소스 매니저 초기화
        # source: int(웹캠 인덱스) 또는 str(로컬 비디오 파일 경로 또는 유튜브 URL)
        self.source = source  # 입력 소스 저장 (숫자형은 웹캠, 문자열은 파일 또는 URL)
        self.cap = None       # OpenCV VideoCapture 객체용 변수 초기화

    def open(self):
        # 비디오 소스를 엽니다. 유튜브 URL일 경우 yt_dlp를 사용하여 실시간 스트림 URL을 추출합니다.
        # 실패 시 RuntimeError 발생
        
        # Case 1: 입력 소스가 정수인 경우 -> 시스템에 연결된 웹캠 오픈
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
            
        # Case 2: 입력 소스가 문자열인 경우 -> 유튜브 URL 또는 로컬 파일 경로 처리
        elif isinstance(self.source, str):
            # 유튜브 링크가 포함되어 있는지 확인
            if "youtube.com" in self.source or "youtu.be" in self.source:
                # yt_dlp 옵션 설정 (가장 좋은 화질의 mp4 포맷 스트림 주소를 무음 모드로 가져옴)
                ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        # 다운로드 없이 비디오 정보 및 실제 스트리밍 미디어 URL 추출
                        info = ydl.extract_info(self.source, download=False)
                        stream_url = info["url"]
                        # 추출된 웹 스트림 URL을 OpenCV에 전달하여 네트워크 비디오 오픈
                        self.cap = cv2.VideoCapture(stream_url)
                    except Exception as e:
                        raise RuntimeError(f"Failed to extract YouTube stream: {e}")
            else:
                # 일반 문자열인 경우 -> 로컬 시스템 내부의 동영상 파일 경로로 간주하여 오픈
                self.cap = cv2.VideoCapture(self.source)
        else:
            raise ValueError("Source must be either int or str")

        # 비디오 장치 또는 파일이 정상적으로 열렸는지 최종 검증
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")

    def read_frame(self) -> tuple[bool, np.ndarray]:
        # 비디오 소스로부터 다음 프레임 이미지를 읽어 반환합니다.
        # 반환값: (성공 여부 bool, 프레임 이미지 numpy array)
        
        # 비디오 인스턴스가 생성되지 않은 예외 상황 방어
        if self.cap is None:
            return False, np.zeros((1, 1, 3), dtype=np.uint8)
        # OpenCV를 통해 실제 1프레임 캡처 수행
        return self.cap.read()

    def release(self):
        # 비디오 캡처 자원을 해제하고 연결을 닫습니다.
        
        # 자원이 활성화되어 있으면 해제 처리 진행
        if self.cap is not None:
            self.cap.release()
            self.cap = None  # 중복 해제 방지를 위해 멤버 변수 초기화
