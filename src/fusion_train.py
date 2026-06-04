# src/fusion_train.py

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from data.dataset import SignLanguageDataset  # dataset.py 파일의 클래스
from data.collate import SignLanguageCollateFn # collate.py 파일의 클래스
from models.model import SignLanguageTranslator # model.py 파일의 클래스

def main():
    # ---------------------------------------------------------
    # 1. 학습 환경 및 하이퍼파라미터 설정
    # ---------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 현재 사용 중인 디바이스: {device}")
    
    # [Sanity Check용 설정] 처음에는 에포크를 2 정도로 짧게 주고 테스트하세요!
    num_epochs = 50
    batch_size = 32
    learning_rate = 5e-5

    # KoBART 토크나이저 초기화
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    
    # ---------------------------------------------------------
    # 2. 데이터셋 및 데이터로더 준비 (Train / Val / Test 분할)
    # ---------------------------------------------------------
    print("📦 데이터 로더를 준비 중입니다...")
    # 주의: json_dir에는 '_merged.json' 파일들이 있는 폴더 경로를 적어주세요.
    full_dataset = SignLanguageDataset(json_dir='./dataset/merged_dataset', mode='sensor_fusion')
    collate_fn = SignLanguageCollateFn(tokenizer=tokenizer)
    
    total_size = len(full_dataset)
    
    # 1차 분할: 전체의 20%를 Test용으로 분리 (Train+Val 8 : Test 2)
    test_size = int(total_size * 0.2)
    temp_train_size = total_size - test_size

    # random_split을 사용하여 무작위 분할 (시드를 고정하여 매번 똑같이 나뉘게 함)
    temp_train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [temp_train_size, test_size], generator=torch.Generator().manual_seed(42) 
    )

    # 2차 분할: 남은 80% 중에서 20%를 Validation용으로 분리 (Train 8 : Val 2)
    val_size = int(temp_train_size * 0.2)
    train_size = temp_train_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        temp_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )   

    print(f"📊 총 데이터: {total_size}개 ➔ Train: {train_size}개 | Val: {val_size}개 | Test: {test_size}개")


    # 각각의 DataLoader 생성
    # 학습용은 데이터를 섞어주고(shuffle=True), 평가용은 섞을 필요가 없습니다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ---------------------------------------------------------
    # 3. 모델 및 옵티마이저 초기화
    # ---------------------------------------------------------
    sample_batch = next(iter(train_loader))
    vision_dim = sample_batch['vision_inputs'].shape[-1]
    sensor_dim = sample_batch['sensor_inputs'].shape[-1]

    print(f"✅ 자동 추출 완료! Vision 차원: {vision_dim} / Sensor 차원: {sensor_dim}")
    print("🤖 AI 모델을 초기화 중입니다...")
    
    model = SignLanguageTranslator(
        vision_dim=vision_dim, 
        sensor_dim=sensor_dim, 
        mode='sensor_fusion', 
        encoder_type='gru' # 💡 'lstm', 'transformer' 로 변경하며 실험 가능!
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # ---------------------------------------------------------
    # 4. 본격적인 학습 루프 (Training Loop) 및 검증 루프
    # ---------------------------------------------------------
    best_val_loss = float('inf') # Train Loss가 아닌 Val Loss 기준으로 최고 성능을 기록
    # Early Stopping 설정
    patience = 10
    early_stop_counter = 0

    # 그래프 출력을 위한 기록용 리스트 생성
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train() # 학습 모드 켬
        total_train_loss = 0.0
        
        # 🟢 [Train Phase] 학습 단계
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for step, batch in enumerate(loop):
            optimizer.zero_grad() # 기울기 초기화
            
            # 4-1. 데이터를 GPU로 이동
            vision_inputs = batch['vision_inputs'].to(device)
            sensor_inputs = batch['sensor_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 4-2. 모델 예측 및 Loss 계산 (KoBART가 내부적으로 CrossEntropyLoss를 계산함)
            outputs = model(vision_inputs, sensor_inputs, attention_mask, labels=labels)
            loss = outputs.loss
            
            # 4-3. 가중치 업데이트 (역전파)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # 진행률 바에 현재 Epoch과 Loss 실시간 출력
            loop.set_postfix(loss=loss.item())
            
        # 1 에포크 평균 Loss 계산
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 🔵 [Validation Phase] 검증 단계 (학습하지 않고 평가만 수행)
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            loop = tqdm(val_loader, leave=True, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
            for step, batch in enumerate(loop):
                v_inputs = batch['vision_inputs'].to(device)
                s_inputs = batch['sensor_inputs'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)

                # 모델 평가 시에는 역전파 (backward) 수행하지 않음
                val_outputs = model(v_inputs, s_inputs, mask, labels=lbls)
                total_val_loss += val_outputs.loss.item()
                loop.set_postfix(val_loss=val_outputs.loss.item())
            
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"✅ Epoch {epoch+1} 종료 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 매 에포크 결과를 리스트에 차곡차곡 저장
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # ---------------------------------------------------------
        # 5. 베스트 모델 가중치 저장
        # ---------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            os.makedirs('./checkpoints', exist_ok=True)

            save_path = f'./checkpoints/best_model_{model.mode}_{model.vision_encoder.encoder_type}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"🎉 최고 성능 갱신! 모델 저장됨 (Val Loss: {best_val_loss:.4f}) ➔ {save_path}\n")
        else:
            early_stop_counter += 1
            print(f"⚠️ Val Loss가 개선되지 않았습니다. Early Stopping 카운트: {early_stop_counter} / {patience}\n")

            if early_stop_counter >= patience:
                print(f"🛑 {patience} 에포크 연속으로 성능이 개선되지 않아 학습을 조기 종료합니다!")
                break # 전체 학습 for 루프 강제 종료

    print("📈 학습이 종료되었습니다. Loss 수렴 그래프를 생성합니다...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss', color='green', marker='o')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='blue', marker='x')
    
    plt.title('Training and Validation Loss Trend', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 프로젝트 폴더 내에 이미지 파일로 저장
    plot_path = './checkpoints/loss_convergence_plot.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 그래프 저장 완료! 확인해 보세요 ➔ {plot_path}")

if __name__ == '__main__':
    main()