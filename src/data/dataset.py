# src/data/dataset.py

# PyTorch Dataset 클래스가 있는 파일 (학습 데이터 로드용)
# 데이터 추출 및 평탄화
# JSON 파일을 하나씩 열어 2D 배열로 되어있는 MediaPipe 좌표와 센서 값을 1D 벡터(평탄화)로 쭉 펴주는 역할
# 순수 비전 모델과 센서 융합 모델을 선택해서 학습할 수 있도록 mode 분기 처리


# 좌표 값만 가지고 학습하는 Baseline 모델을 먼저 학습 시킨 다음에 velocity 정보를 넣고 학습시켜보기

import os
import glob
import json
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, json_dir, mode='sensor_fusion'):
        """
        :param file_paths: 정규화 및 병합이 완료된 JSON 파일 경로 리스트
        :param mode: 'vision_only' (카메라 손 좌표 사용) 또는 'sensor_fusion' (장갑 센서 사용)
        """
        self.mode = mode
        self.file_paths = glob.glob(os.path.join(json_dir, "*_merged.json"))

        # 파일이 비어있으면 에러 띄우기 (경로 설정 실수 방지)
        if len(self.file_paths) == 0:
            raise ValueError(f"🚨 '{json_dir}' 경로에 병합된 데이터가 없습니다! 경로를 확인하세요.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 정답 한국어 텍스트 추출
        korean_text = data['metadata'].get('koreanText', '')

        pose_vision_frames = []
        face_vision_frames = []
        left_hand_vision_frames = []
        right_hand_vision_frames = []
        both_hand_vision_frames = []
        sensor_frames = []

        # 2. 프레임 순회
        for frame in data['frames']:
            # --- [공통] 비수지 기호 및 포즈 평탄화 (Flatten) ---
            # 2D 배열 [[x, y, z], [x, y, z]] 을 1D 배열 [x, y, z, x, y, z] 로 변환
            pose = sum(frame.get('pose_keypoints', []), [])
            face = sum(frame.get('face_keypoints', []), [])
            lip = sum(frame.get('lip_keypoints', []), [])
            eyebrow = sum(frame.get('eyebrow_keypoints', []), [])
            eye = sum(frame.get('eye_keypoints', []), [])

            # 비수지 기호 벡터 결합
            pose_feat = pose
            face_feat = face + lip + eyebrow + eye
            pose_vision_frames.append(pose_feat)
            face_vision_frames.append(face_feat)

            # --- [분기] 수지 기호 (손 모양) 데이터 할당 ---
            if self.mode == 'vision_only':
                # MediaPipe 카메라 손 좌표 사용
                left_h = sum(frame.get('left_hand_keypoints', []), [])
                right_h = sum(frame.get('right_hand_keypoints', []), [])
                
                # 결측치 방어 (21관절 * 3축 = 63)
                left_h = left_h if left_h else [0.0] * 63
                right_h = right_h if right_h else [0.0] * 63

                # 비전 텐서에 모두 결합, 센서 텐서는 사용
                left_hand_vision_frames.append(left_h)
                right_hand_vision_frames.append(right_h)
                both_hand_vision_frames.append([0.0] * 126)
                sensor_frames.append([0.0] * 28)

            elif self.mode == 'sensor_fusion':
                left_h = sum(frame.get('left_hand_keypoints', []), [])
                right_h = sum(frame.get('right_hand_keypoints', []), [])
                
                # 결측치 방어 (21관절 * 3축 = 63)
                left_h = left_h if left_h else [0.0] * 63
                right_h = right_h if right_h else [0.0] * 63
                both_hand_vision_frames.append(left_h + right_h)

                # 센서 텐서에 장갑 데이터(정규화됨) 할당
                left_s = frame.get('sensor_left_hand', {"flex_mcp_pip": [0.0]*10, "imu_rpy": [0.0]*4})
                right_s = frame.get('sensor_right_hand', {"flex_mcp_pip": [0.0]*10, "imu_rpy": [0.0]*4})

                left_feat = left_s['flex_mcp_pip'] + left_s['imu_rpy']
                right_feat = right_s['flex_mcp_pip'] + right_s['imu_rpy']

                sensor_frames.append(left_feat + right_feat)
                left_hand_vision_frames.append([0.0] * 63)
                right_hand_vision_frames.append([0.0] * 63)

        # 3. 파이토치 FloatTensor로 변환
        pose_vision_tensor = torch.FloatTensor(pose_vision_frames)
        face_vision_tensor = torch.FloatTensor(face_vision_frames)
        left_hand_vision_tensor = torch.FloatTensor(left_hand_vision_frames)
        right_hand_vision_tensor = torch.FloatTensor(right_hand_vision_frames)
        both_hand_vision_tensor = torch.FloatTensor(both_hand_vision_frames)
        sensor_tensor = torch.FloatTensor(sensor_frames)

        return pose_vision_tensor, face_vision_tensor, left_hand_vision_tensor, right_hand_vision_tensor, both_hand_vision_tensor, sensor_tensor, korean_text