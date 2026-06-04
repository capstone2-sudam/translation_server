# src/data/collate.py

# PyTorch collate_fn 함수가 있는 파일 (가변 길이 패딩용)
# 서로 다른 영상 길이를 하나의 Batch로 합치기 위해 가장 

# 가장 긴 프레임에 맞춰 빈 공간을 0으로 채우는(Zero-Padding) 핵심 역할을 합니다.


import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

class SignLanguageCollateFn:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        # KoBART 토크나이저를 주입받아 사용합니다.
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        batch: dataset.__getitem__ 에서 반환된 (vision, sensor, text) 튜플의 리스트
        """
        pose_seqs = [item[0] for item in batch]
        face_seqs = [item[1] for item in batch]
        left_seqs = [item[2] for item in batch]
        right_seqs = [item[3] for item in batch]
        both_seqs = [item[4] for item in batch]
        sensor_seqs = [item[5] for item in batch]
        texts = [item[6] for item in batch]

        # 1. 시계열 패딩 (가변 길이 맞추기)
        # batch_first=True 를 주어 차원을 (Batch, Seq_len, Feature_dim) 으로 고정합니다.
        # batch: 한 번에 학습할 데이터(비디오/수어 문장)의 개수
        # Seq_len: 하나의 비디오가 가진 프레임 수 (시퀀스 길이)
        # Feature_dim: 트특징 차원으로 1개 프레임 안에 들어있는 관절 좌표와 센서 값의 총 개수

        pose_padded = torch.nn.utils.rnn.pad_sequence(pose_seqs, batch_first=True)
        face_padded = torch.nn.utils.rnn.pad_sequence(face_seqs, batch_first=True)
        left_padded = torch.nn.utils.rnn.pad_sequence(left_seqs, batch_first=True)
        right_padded = torch.nn.utils.rnn.pad_sequence(right_seqs, batch_first=True)
        both_padded = torch.nn.utils.rnn.pad_sequence(both_seqs, batch_first=True)
        sensor_padded = torch.nn.utils.rnn.pad_sequence(sensor_seqs, batch_first=True)

        # 2. 어텐션 마스크 (Attention Mask) 생성
        # 모델(Transformer)이 패딩된 0.0 값을 실제 데이터로 착각하지 않도록,
        # 실제 데이터가 있는 부분은 1, 패딩된 빈 공간은 0으로 표시하는 마스크를 만듭니다.
        seq_lengths = [len(seq) for seq in pose_seqs] # 원본 비디오들의 진짜/실제 프레임 길이가 몇 개였는 지 각각 기록 해두기
        # 패딩이 완료된 텐서의 최대 길이(Max_len)
        max_len = pose_padded.size(1)
        # 일단 전부 0으로 채워진 도화지(마스크)를 만듦 - 크기는 (배치 수, 최대 프레임 수)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        
        # 진짜 데이터가 있던 길이(seq_lengths)만큼만 1로 바꿔줌
        for i, length in enumerate(seq_lengths):
            attention_mask[i, :length] = 1

        # 3. 텍스트 정답 라벨(Label) 토크나이징
        wrapped_texts = [f"<s> {text}</s>" for text in texts]

        encoded_texts = self.tokenizer(
            wrapped_texts,
            padding=True,         # 가장 긴 문장에 맞춰 패딩
            truncation=True,      # 너무 긴 문장은 자르기
            max_length=128,
            return_tensors="pt"   # 파이토치 텐서로 반환
        )

        # 4. Loss 계산에서 패딩 토큰 무시 처리
        # NLP 모델 학습의 정석: 패딩된 토큰 아이디는 오차(Loss) 계산에서 제외하기 위해 -100 으로 치환합니다.
        labels = encoded_texts["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 최종적으로 모델의 forward 함수에 쏙 들어갈 딕셔너리 포맷으로 반환합니다.
        return {
        'pose_inputs': pose_padded,
        'face_inputs': face_padded,
        'left_hand_inputs': left_padded,
        'right_hand_inputs': right_padded,
        'hand_vision_inputs': both_padded,
        'sensor_inputs': sensor_padded,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': texts
    }