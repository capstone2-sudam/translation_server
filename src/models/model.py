# src/models/model.py

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import math
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_type='gru', num_layers=2, dropout = 0.2, max_len=512):
        super().__init__()
        self.encoder_type = encoder_type
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)  

        rnn_dropout = dropout if num_layers > 1 else 0.0
                # 실험할 모델(인코더) 종류 선택
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, 
                               hidden_size=hidden_dim // 2, 
                               num_layers=num_layers, 
                               batch_first=True, 
                               bidirectional=True, 
                               dropout=rnn_dropout)
            
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(input_size=hidden_dim, 
                              hidden_size=hidden_dim // 2, 
                              num_layers=num_layers, 
                              batch_first=True, 
                              bidirectional=True,
                              dropout=rnn_dropout)
            
        elif encoder_type == 'transformer':
            self.pos_encoder = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, attention_mask=None):
        # x shape: (Batch, Seq_len, input_dim)
        x = self.input_projection(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        
        if self.encoder_type in ['lstm', 'gru']:
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).cpu()
                packed = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
                packed_out, _ = self.rnn(packed)
                outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            else:
                outputs, _ = self.rnn(x)

        elif self.encoder_type == 'transformer':
            x = self.pos_encoder(x)
            if attention_mask is not None:
                src_key_padding_mask = ~attention_mask.bool()
                outputs = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            else:
                outputs = self.transformer(x)
        return outputs # (Batch, Seq_len, hidden_dim) 반환


class StreamFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.block = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),          
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.proj(x)
        # x = self.norm(x)
        return x + self.block(self.norm(x))

class SignLanguageTranslator(nn.Module):
    def __init__(self, pose_dim, face_dim, left_dim, right_dim, both_dim, sensor_dim=28, mode='sensor_fusion', encoder_type='gru', dropout=0.2):
        super().__init__()
        self.mode = mode
        
        # KoBART 모델 불러오기 (우리는 여기서 디코더 파트만 사용합니다)
        self.kobart = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        # KoBART가 요구하는 입력 차원 크기 (보통 768 차원)
        kobart_dim = self.kobart.config.d_model 

        if mode == 'sensor_fusion':
            stream_dim = 256
            self.pose_encoder = ModalityEncoder(pose_dim, stream_dim, encoder_type, dropout=dropout)
            self.face_encoder = ModalityEncoder(face_dim, stream_dim, encoder_type, dropout=dropout)            
            self.hand_encoder = ModalityEncoder(both_dim, stream_dim, encoder_type, dropout=dropout)
            self.sensor_encoder = ModalityEncoder(sensor_dim, stream_dim, encoder_type, dropout=dropout)
            fusion_input_dim = stream_dim * 4
            
        elif mode == 'vision_only':
            stream_dim = 256
            self.pose_encoder = ModalityEncoder(pose_dim, stream_dim, encoder_type, dropout=dropout)
            self.face_encoder = ModalityEncoder(face_dim, stream_dim, encoder_type, dropout=dropout)            
            self.left_encoder = ModalityEncoder(left_dim, stream_dim, encoder_type, dropout=dropout)
            self.right_encoder = ModalityEncoder(right_dim, stream_dim, encoder_type, dropout=dropout)
            fusion_input_dim = stream_dim * 4

        self.fusion_layer = StreamFusion(
            fusion_input_dim,
            kobart_dim
        )   

    def encode_inputs(self, pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask):
        if self.mode == 'sensor_fusion':
            # 카메라와 장갑 데이터를 각각의 인코더에 통과시킵니다.
            pose_feat = self.pose_encoder(pose_inputs, attention_mask) # 결과: (Batch, Seq_len, 192)
            face_feat = self.face_encoder(face_inputs, attention_mask) 
            hand_feat = self.hand_encoder(both_inputs, attention_mask) 
            sensor_feat = self.sensor_encoder(sensor_inputs, attention_mask) 
        
            # 두 특징을 마지막 차원(dim=-1) 기준으로 이어 붙입니다.
            fused = torch.cat([pose_feat, face_feat, hand_feat, sensor_feat], dim=-1)
            
        elif self.mode == 'vision_only':
            pose_feat = self.pose_encoder(pose_inputs, attention_mask) # 결과: (Batch, Seq_len, 192)
            face_feat = self.face_encoder(face_inputs, attention_mask)
            left_feat = self.left_encoder(left_inputs, attention_mask) 
            right_feat = self.right_encoder(right_inputs, attention_mask)

            # 두 특징을 마지막 차원(dim=-1) 기준으로 이어 붙입니다.
            fused = torch.cat([pose_feat, face_feat, left_feat, right_feat], dim=-1)

        encoder_outputs = self.fusion_layer(fused)
        # encoder_outputs = (encoder_outputs * attention_mask.unsqueeze(-1).float())
        
        return encoder_outputs

    def forward(self, pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask, labels=None):
        encoder_hidden_states = self.encode_inputs(pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        outputs = self.kobart(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(self, pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask, max_length=50, num_beams=5):
        encoder_hidden_states = self.encode_inputs(pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        generated_ids = self.kobart.generate(
            encoder_outputs = encoder_outputs,
            attention_mask = attention_mask,
            max_length = max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        return generated_ids
    