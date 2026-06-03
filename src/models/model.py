# src/models/model.py

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

# ---------------------------------------------------------
# 1. 단일 모달리티 인코더 (각각의 데이터를 처리하는 전문가)
# ---------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_type='gru', num_layers=2):
        super().__init__()
        self.encoder_type = encoder_type
        
        # 입력 차원을 모델이 계산하기 좋은 Hidden 차원으로 뻥튀기 해줍니다.
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        drop_rate = 0.2 if num_layers > 1 else 0.0
        
        # 실험할 모델(인코더) 종류 선택 - GRU/LSTM은 2 ~ 3정도로 설정해보기
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, batch_first=True, bidirectional=True, dropout=drop_rate)
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, num_layers, batch_first=True, bidirectional=True, dropout=drop_rate)
        elif encoder_type == 'transformer':
            # 트랜스포머는 RNN과 달리 층을 깊게 쌓을수록 문맥을 훨씬 더 잘 이해하도록 설계된 구조 (num_layers=4 정도가 가장 속도 대비 성능이 좋음. transformer 사용 시 4나 6정도로 설정해서 돌려보기)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=0.2, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (Batch, Seq_len, input_dim)
        x = self.relu(self.input_projection(x))
        
        if self.encoder_type in ['lstm', 'gru']:
            outputs, _ = self.rnn(x) 
            # 양방향(Bidirectional)이므로 결과가 (hidden_dim // 2) * 2 = hidden_dim 으로 딱 맞춰집니다.
        elif self.encoder_type == 'transformer':
            outputs = self.transformer(x)
            
        return outputs # (Batch, Seq_len, hidden_dim) 반환


# ---------------------------------------------------------
# 2. 듀얼 인코더 & KoBART 디코더 통합 아키텍처
# ---------------------------------------------------------
class SignLanguageTranslator(nn.Module):
    # 
    def __init__(self, vision_dim, sensor_dim=26, mode='sensor_fusion', encoder_type='gru'):
        """
        vision_dim: 비전 특징 벡터의 길이 (예: 얼굴+포즈+입술 = 약 100~200 차원)
        sensor_dim: 양손 센서 특징 벡터의 길이 (기본값 26 = (10+3) * 2)
        """
        super().__init__()
        self.mode = mode
        
        # KoBART 모델 불러오기 (우리는 여기서 디코더 파트만 사용합니다)
        self.kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        # KoBART가 요구하는 입력 차원 크기 (보통 768 차원)
        kobart_dim = self.kobart.config.d_model 
        
        if mode == 'sensor_fusion':
            # 💡 [핵심] 듀얼 인코더 구조!
            # 나중에 두 개를 합쳤을 때 768이 되도록, 각각의 인코더는 384 차원으로 출력하게 만듭니다.
            half_dim = kobart_dim // 2 
            self.vision_encoder = ModalityEncoder(vision_dim, half_dim, encoder_type)
            self.sensor_encoder = ModalityEncoder(sensor_dim, half_dim, encoder_type)
            
        elif mode == 'vision_only':
            # 센서가 없을 때는 비전 인코더 하나가 768 차원을 모두 담당합니다.
            self.vision_encoder = ModalityEncoder(vision_dim, kobart_dim, encoder_type)

    def forward(self, vision_inputs, sensor_inputs, attention_mask, labels=None):
        
        # 1. 인코더 통과 및 데이터 퓨전(Fusion)
        if self.mode == 'sensor_fusion':
            # 카메라와 장갑 데이터를 각각의 인코더에 통과시킵니다.
            v_feat = self.vision_encoder(vision_inputs) # 결과: (Batch, Seq_len, 384)
            s_feat = self.sensor_encoder(sensor_inputs) # 결과: (Batch, Seq_len, 384)
            
            # 두 특징을 마지막 차원(dim=-1) 기준으로 이어 붙입니다.
            # 384 + 384 = 768 (KoBART가 딱 좋아하는 사이즈 완성!)
            encoder_outputs = torch.cat([v_feat, s_feat], dim=-1)
            
        elif self.mode == 'vision_only':
            encoder_outputs = self.vision_encoder(vision_inputs) # 결과: (Batch, Seq_len, 768)

        # 2. KoBART 디코더 텍스트 생성 및 학습 (오차 계산)
        # HuggingFace 모델은 encoder_outputs를 튜플 형태로 받기를 원하므로 ( ,) 로 감싸줍니다.
        outputs = self.kobart(
            encoder_outputs=(encoder_outputs,),
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 학습 중일 때는 loss를 반환하고, 추론 중일 때는 디코딩된 결과인 logits를 반환합니다.
        return outputs