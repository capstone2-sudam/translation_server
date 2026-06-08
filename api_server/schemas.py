from pydantic import BaseModel
from typing import List

# 1. 프레임 단일 데이터 스펙 (메인 백엔드의 AlignedSignDTO와 매칭)
class FrameData(BaseModel):
    timestamp: int
    left_sensor_payload: List[float]
    right_sensor_payload: List[float]
    frame_number: int
    face_keypoints: List[List[float]]
    lip_keypoints: List[List[float]]
    eyebrow_keypoints: List[List[float]]
    eye_keypoints: List[List[float]]
    pose_keypoints: List[List[float]]
    left_hand_keypoints: List[List[float]]
    right_hand_keypoints: List[List[float]]

# 2. 클라이언트(메인 백엔드)에서 최종적으로 보내는 JSON 구조 ({"frames": [...]})
class TranslationRequest(BaseModel):
    frames: List[FrameData]