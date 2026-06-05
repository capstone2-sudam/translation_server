# src/models/customeAttention.py

import torch
import torch.nn as nn
import math

class CustomKeypointAttention(nn.Module):
    def __init__(self, num_keypoints, in_channels=3, hidden_dim=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # 1. 차원 축소/확장을 위한 선형 변환 (Q, K, V 생성용)
        self.q_linear = nn.Linear(in_channels, hidden_dim)
        self.k_linear = nn.Linear(in_channels, hidden_dim)
        self.v_linear = nn.Linear(in_channels, hidden_dim)
        
        # 2. 💡 논문의 핵심 1: Softmax 대신 Tanh 사용 (음의 상관관계 학습) 
        self.activation = nn.Tanh()
        
        # 3. 💡 논문의 핵심 2: Spatial Global Regularization (SGR) 
        # 모든 데이터에 공통적으로 적용되는 관절 간의 전역적 관계 (N x N 학습가능한 파라미터)
        self.sgr = nn.Parameter(torch.zeros(num_keypoints, num_keypoints))
        nn.init.xavier_uniform_(self.sgr) # 초기화
        
        # 출력 변환용
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Num_keypoints, In_channels)
        # 예: (32, 100, 42, 3) -> 배치가 32, 100프레임, 양손 관절 42개, XYZ 3차원
        
        B, T, N, C = x.size()
        
        # Q, K, V 투영
        Q = self.q_linear(x) # (B, T, N, hidden_dim)
        K = self.k_linear(x) # (B, T, N, hidden_dim)
        V = self.v_linear(x) # (B, T, N, hidden_dim)
        
        # Q와 K의 전치 행렬 곱 (공간적 Attention Map 생성)
        # K.transpose: (B, T, hidden_dim, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        
        # 💡 Softmax 대신 Tanh 적용 
        attn_map = self.activation(attn_scores) # shape: (B, T, N, N)
        
        # 💡 SGR 더하기 (전역 정규화) 
        # sgr은 (N, N) 차원이지만 브로드캐스팅을 통해 (B, T, N, N)에 더해짐
        attn_map = attn_map + self.sgr
        
        # Attention Map과 V 곱하기
        out = torch.matmul(attn_map, V) # (B, T, N, hidden_dim)
        
        # 마지막 비선형 활성화 및 잔차 연결(Residual Connection)
        out = self.leaky_relu(self.out_linear(out))
        
        return out
    
    '''
    [Phase 1] 베이스라인 완성 (현재 단계)할 일: 
    앞서 우리가 완성한 GRU 기반 ModalityEncoder (Flatten 데이터 사용)로 먼저 전체 파이프라인이 정상 작동하는지 학습을 돌립니다.
    목표: Loss가 떨어지고 KoBART가 번역을 생성하는지 확인 (이 기준점이 있어야 나중에 논문 기법을 넣었을 때 성능이 올랐는지 비교할 수 있습니다).
    
    [Phase 2] 커스텀 Attention 모듈 이식 (논문 아키텍처 적용)할 일: 
    GRU 대신 논문에서 제안한 Tanh + SGR(공간 전역 정규화)가 적용된 커스텀 Attention 인코더로 교체합니다.  
    방법: model.py에 아래의 커스텀 클래스를 추가하여 사용합니다.
    
    [Phase 3] 학습 방법론 이식 (자가 증류 - Self Distillation)할 일: 
    단순히 4개 스트림을 합치는 것(Fuse Head)을 넘어, 각 스트림(손, 얼굴 등)이 서로의 지식을 배울 수 있도록 
    자가 증류(Self-Distillation) Loss를 학습 코드(train.py)에 추가합니다.  
    '''