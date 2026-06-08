# api_server/service.py
import torch
import numpy as np
from schemas import FrameData
from typing import List
from transformers import PreTrainedTokenizerFast
from src.models.model import SignLanguageTranslator # 기존에 작성하신 모델 클래스 임포트


# ---------------------------------------------------------
# 1. 전처리 함수 (학습 파일 dataset.py와 100% 동일)
# ---------------------------------------------------------
def apply_hand_local_normalization(hand_sequence: np.ndarray) -> np.ndarray:
    out_seq = np.zeros_like(hand_sequence)
    valid_mask = ~np.all(np.isclose(hand_sequence, 0.0, atol=1e-6), axis=(1, 2))
    
    if np.any(valid_mask):
        valid_hands = hand_sequence[valid_mask] # (V, 21, 3)
        wrists = valid_hands[:, 0:1, :]         # (V, 1, 3)
        mid_bases = valid_hands[:, 9:10, :]     # (V, 1, 3)
        
        shifted = valid_hands - wrists
        scales = np.linalg.norm(mid_bases - wrists, axis=-1, keepdims=True) # (V, 1, 1)
        scales[scales < 1e-6] = 1.0 # 0 나누기 방지
        
        out_seq[valid_mask] = shifted / scales
    return out_seq

def add_velocity(frames_arr: np.ndarray) -> np.ndarray:
    vel = np.diff(frames_arr, axis=0, prepend=frames_arr[:1])
    is_zero = np.all(np.isclose(frames_arr, 0.0, atol=1e-6), axis=-1, keepdims=True)
    is_prev_zero = np.pad(is_zero[:-1], ((1, 0), (0, 0)), constant_values=False)
    mask = is_zero | is_prev_zero
    vel = np.where(mask, 0.0, vel)
    return np.concatenate([frames_arr, vel], axis=-1)

# ---------------------------------------------------------
# 2. 모델 차원 설정 상수화 (안전한 아키텍처)
# ---------------------------------------------------------
MODEL_DIM_CONFIG = {
    "pose_dim": 24, # velocity 미적용
    "face_dim": 273, # velocity 미적용
    "left_dim": 126, # 63 + velocity = 126
    "right_dim": 126, # 63 + velocity = 126
    "both_dim": 252, # left(126) + right(126) = 252
    "sensor_dim": 28, # left(14) + right(14) = 28
}

class TranslationService:
    def __init__(self):
        print("⏳ AI 모델 로딩 중... 잠시만 기다려주세요.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        
        # ==========================================
        # [A] Sensor Fusion 모델 로드
        # ==========================================
        self.model_fusion = SignLanguageTranslator(**MODEL_DIM_CONFIG, mode='sensor_fusion', encoder_type='gru', dropout=0.3).to(self.device) 
        
        # sensor_fusion 모델의 실제 pth 경로 넣기
        fusion_ckpt = torch.load('./checkpoints/best_model_sensor_fusion_gru_lr_3e-05_dropout_0.3_frozen_enc.pth', map_location=self.device)
        self.model_fusion.load_state_dict(fusion_ckpt['model_state_dict'])
        self.model_fusion.eval()

        # ==========================================
        # 2. Vision Only 모델 (동일한 설정값 공유)
        # ==========================================
        self.model_vision = SignLanguageTranslator(**MODEL_DIM_CONFIG, mode='vision_only', encoder_type='gru', dropout=0.3).to(self.device) 

        # vision_only 모델의 실제 pth 경로 넣기
        vision_ckpt = torch.load('./checkpoints/best_model_vision_only_gru_lr_3e-05_dropout_0.3_frozen_enc.pth', map_location=self.device)
        self.model_vision.load_state_dict(vision_ckpt['model_state_dict'])
        self.model_vision.eval()

        print("✅ 앙상블(Vision + Fusion) AI 모델 로딩 완료! 추론 서버가 준비되었습니다.")

    @torch.no_grad()
    def infer(self, frames_data: list) -> str:
        """
        넘어온 JSON 데이터를 학습 때와 똑같이 Numpy 정규화/속도 계산 후 모델에 입력합니다.
        """
        
        pose_list, face_list, sensor_list = [], [], []
        left_hand_raw, right_hand_raw = [], []
        
        # 1. 데이터 파싱 (평탄화 및 Shape 맞추기)
        for pydantic_frame in frames_data:
            frame = pydantic_frame.model_dump()
            # 포즈 및 얼굴 (1D Flatten)
            pose = sum(frame.get('pose_keypoints', []), [])
            face = (sum(frame.get('face_keypoints', []), []) +
                    sum(frame.get('lip_keypoints', []), []) +
                    sum(frame.get('eyebrow_keypoints', []), []) +
                    sum(frame.get('eye_keypoints', []), []))
            
            pose_list.append(pose)
            face_list.append(face)

            # 센서 (14 + 14 = 28차원 패딩 방어)
            ls = frame.get('left_sensor_payload') or []
            rs = frame.get('right_sensor_payload') or []
            if len(ls) < 14: ls += [0.0] * (14 - len(ls))
            if len(rs) < 14: rs += [0.0] * (14 - len(rs))
            sensor_list.append(ls[:14] + rs[:14])

            # 손 (로컬 정규화를 위해 원본 21x3 배열 유지)
            lh_raw = frame.get('left_hand_keypoints', [])
            rh_raw = frame.get('right_hand_keypoints', [])
            
            lh_arr = np.array(lh_raw, dtype=np.float32) if lh_raw else np.zeros((21, 3), dtype=np.float32)
            rh_arr = np.array(rh_raw, dtype=np.float32) if rh_raw else np.zeros((21, 3), dtype=np.float32)
            
            if lh_arr.shape != (21, 3): lh_arr = np.zeros((21, 3), dtype=np.float32)
            if rh_arr.shape != (21, 3): rh_arr = np.zeros((21, 3), dtype=np.float32)

            left_hand_raw.append(lh_arr)
            right_hand_raw.append(rh_arr)

        # 2. NumPy 배열 변환 및 시퀀스 단위 정규화/속도 추가 (학습과 100% 동일)
        pose_arr = np.array(pose_list, dtype=np.float32)
        face_arr = np.array(face_list, dtype=np.float32)
        sensor_arr = np.array(sensor_list, dtype=np.float32)

        left_seq = np.stack(left_hand_raw, axis=0)   # (T, 21, 3)
        right_seq = np.stack(right_hand_raw, axis=0) # (T, 21, 3)

        # 로컬 정규화
        left_seq = apply_hand_local_normalization(left_seq)
        right_seq = apply_hand_local_normalization(right_seq)

        # 평탄화 (T, 63)
        left_flat = left_seq.reshape(len(left_seq), -1)
        right_flat = right_seq.reshape(len(right_seq), -1)

        # 속도 추가 (T, 126)
        left_flat = add_velocity(left_flat)
        right_flat = add_velocity(right_flat)

        T = len(pose_arr)

        # 3. 모델별 입력 텐서 준비 (분기 처리)
        # --- Fusion 모델용 텐서 ---
        both_flat = np.concatenate([left_flat, right_flat], axis=-1) # (T, 252)
        zero_126 = np.zeros((T, 126), dtype=np.float32)
        
        # --- Vision 모델용 텐서 ---
        both_zero = np.zeros((T, 252), dtype=np.float32)
        sensor_zero = np.zeros((T, 28), dtype=np.float32)

        # 4. 파이토치 텐서 변환 및 Batch(1) 차원 추가
        def to_tensor(arr): return torch.FloatTensor(arr).unsqueeze(0).to(self.device)

        p_t = to_tensor(pose_arr)
        f_t = to_tensor(face_arr)
        
        # Fusion Input
        l_fus, r_fus = to_tensor(zero_126), to_tensor(zero_126)
        b_fus, s_fus = to_tensor(both_flat), to_tensor(sensor_arr)
        
        # Vision Input
        l_vis, r_vis = to_tensor(left_flat), to_tensor(right_flat)
        b_vis, s_vis = to_tensor(both_zero), to_tensor(sensor_zero)

        # 마스크 생성 (패딩이 없으므로 전부 1)
        mask = torch.ones(1, p_t.size(1), dtype=torch.long).to(self.device)

        # 5. 모델 추론 및 앙상블 점수 비교
        # [A] Fusion 추론
        out_fusion = self.model_fusion.generate(
            p_t, f_t, l_fus, r_fus, b_fus, s_fus, mask, 
            max_length=50, num_beams=5, return_dict_in_generate=True, output_scores=True
        )
        score_fusion = out_fusion.sequences_scores[0].item()
        text_fusion = self.tokenizer.decode(out_fusion.sequences[0], skip_special_tokens=True)

        # [B] Vision 추론
        out_vision = self.model_vision.generate(
            p_t, f_t, l_vis, r_vis, b_vis, s_vis, mask, 
            max_length=50, num_beams=5, return_dict_in_generate=True, output_scores=True
        )
        score_vision = out_vision.sequences_scores[0].item()
        text_vision = self.tokenizer.decode(out_vision.sequences[0], skip_special_tokens=True)

        print(f"[Ensemble] Fusion Score: {score_fusion:.4f} | Text: {text_fusion}")
        print(f"[Ensemble] Vision Score: {score_vision:.4f} | Text: {text_vision}")

        # 점수 비교 (0에 가까울수록 확률 높음)
        if score_fusion >= score_vision:
            return text_fusion
        else:
            return text_vision

translation_service = TranslationService()