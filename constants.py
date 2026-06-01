"""
constants.py
YOLO11n-pose 기반 포즈 분석 시스템 공통 상수 및 상태 정의 파일
"""

from enum import Enum

# ==========================================
# 1. 의존성 안내 (설치 명령어)
# pip install ultralytics opencv-python yt-dlp numpy
# ==========================================

# ==========================================
# 2. 초기화 관련 상수
# ==========================================
INIT_DURATION        = 7.0     # 초기화 데이터 수집 시간 (초)
ABSENCE_THRESHOLD    = 3.0     # 사용자 이탈 판정 기준 시간 (초)
ALPHA                = 0.8     # 안전 보폭 비율 (다리 길이 대비)
CONF_THRESHOLD       = 0.5     # 키포인트 신뢰도 최소값
MIN_BBOX_AREA        = 5000.0  # 유효 바운딩박스 최소 면적 (픽셀²)
LEG_LENGTH_TOLERANCE = 0.15    # 다리 길이 불일치 허용 오차 비율 (±15%)
MISMATCH_THRESHOLD   = 1.5     # 사용자 불일치(ID/다리길이) 감지 후 리셋 대기 시간 (초)

# ==========================================
# 3. YOLO 키포인트 인덱스 (COCO 17 keypoints 기준)
# 각 키포인트는 (x, y, confidence) 형태로 제공됨
# ==========================================
KP_LEFT_HIP    = 11
KP_RIGHT_HIP   = 12
KP_LEFT_KNEE   = 13
KP_RIGHT_KNEE  = 14
KP_LEFT_ANKLE  = 15
KP_RIGHT_ANKLE = 16

# ==========================================
# 4. 시각화 색상 (BGR 형식)
# ==========================================
COLOR_HIP_ANKLE = (0, 255, 255)   # 노란색: 엉덩이·발목
COLOR_KNEE      = (0, 255, 0)     # 초록색: 무릎
COLOR_SKELETON  = (255, 255, 255) # 흰색: 연결선
COLOR_LEG_LINE  = (0, 255, 255)   # 노란색: 다리 길이 강조선
COLOR_INIT_TEXT = (0, 255, 0)     # 초록색: 초기화 중 텍스트
COLOR_DONE_TEXT = (255, 200, 0)   # 주황색: 완료 텍스트
COLOR_BBOX      = (0, 200, 255)   # 주황색: 바운딩박스
COLOR_BAR_BG    = (100, 100, 100) # 회색: 진행률 바 배경
COLOR_BAR_FG    = (0, 200, 255)   # 파란색: 진행률 바 전경

# ==========================================
# 5. 상태 Enum 정의
# ==========================================
class InitState(Enum):
    NONE        = "NONE"         # 초기 상태 또는 리셋 직후
    IN_PROGRESS = "IN_PROGRESS"  # 초기화 진행 중 (데이터 누적)
    DONE        = "DONE"         # 초기화 완료
