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
    
    num_epochs = 50
    batch_size = 32
    learning_rate = 5e-5
    dropout = 0.2

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
        full_dataset, [temp_train_size, test_size], generator=torch.Generator().manual_seed(42))

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
    # dataset[0]은 (pose, face, left, right, both, sensor, text) 튜플을 반환함
    sample_data = full_dataset[0]
    dim_config = {
        "pose_dim": sample_data[0].shape[-1],
        "face_dim": sample_data[1].shape[-1],
        "left_dim": sample_data[2].shape[-1],
        "right_dim": sample_data[3].shape[-1],
        "both_dim": sample_data[4].shape[-1],
        "sensor_dim": sample_data[5].shape[-1],
    }
    print(f"✅ 자동 추출 완료!")
    print(f"🔍 자동 추출된 차원 정보: {dim_config}")
    print("🤖 AI 모델을 초기화 중입니다...")
    
    model = SignLanguageTranslator(
        **dim_config,
        mode='sensor_fusion', 
        encoder_type='gru', # 💡 'lstm', 'transformer' 로 변경하며 실험 가능!
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True
    )

    # ---------------------------------------------------------
    # 4. 본격적인 학습 루프 (Training Loop) 및 검증 루프
    # ---------------------------------------------------------
    best_val_loss = float('inf') # Train Loss가 아닌 Val Loss 기준으로 최고 성능을 기록
    # Early Stopping 설정
    patience = 10
    early_stop_counter = 0

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
            pose_inputs = batch['pose_inputs'].to(device)
            face_inputs = batch['face_inputs'].to(device)
            left_inputs = batch['left_hand_inputs'].to(device)
            right_inputs = batch['right_hand_inputs'].to(device)
            both_inputs = batch['hand_vision_inputs'].to(device)
            sensor_inputs = batch['sensor_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
                        
            # 모델 입력 파라미터 확인 필요
            outputs = model(pose_inputs, face_inputs, left_inputs, right_inputs, both_inputs, sensor_inputs, attention_mask, labels=labels)
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
            for batch in val_loader:
                p_inputs = batch['pose_inputs'].to(device)
                f_inputs = batch['face_inputs'].to(device)
                l_inputs = batch['left_hand_inputs'].to(device)
                r_inputs = batch['right_hand_inputs'].to(device)
                b_inputs = batch['hand_vision_inputs'].to(device)
                s_inputs = batch['sensor_inputs'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)

                # 모델 평가 시에는 역전파 (backward) 수행하지 않음
                val_outputs = model(p_inputs, f_inputs, l_inputs, r_inputs, b_inputs, s_inputs, mask, labels=lbls)
                total_val_loss += val_outputs.loss.item()
                loop.set_postfix(val_loss=val_outputs.loss.item())
            
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"✅ Epoch {epoch+1} 종료 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        with torch.no_grad():
            sample_p = p_inputs[0:1]
            sample_f = f_inputs[0:1]
            sample_l = l_inputs[0:1]
            sample_r = r_inputs[0:1]
            sample_b = b_inputs[0:1]
            sample_s = s_inputs[0:1]
            sample_mask = mask[0:1]
            sample_label = lbls[0].clone()

            # 생성(Generate) 함수 호출
            generated_ids = model.generate(
                sample_p, sample_f, sample_l, sample_r, sample_b, sample_s, sample_mask, max_length=50
            )

            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            sample_label[sample_label == -100] = tokenizer.pad_token_id
            true_text = tokenizer.decode(sample_label.cpu(), skip_special_tokens=True)

            print(f"\n[👀 Epoch {epoch+1} 번역 테스트]")
            print(f"🎯 정답: {true_text}")
            print(f"🤖 예측: {pred_text}\n")
            
        scheduler.step(avg_val_loss)

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

            save_path = f'./checkpoints/best_model_{model.mode}_{model.pose_encoder.encoder_type}_{learning_rate}_dropout_{dropout}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)

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
    plot_dir = './checkpoints/loss_plot'
    os.makedirs(plot_dir, exist_ok=True) # 💡 폴더 생성 방어 코드 추가
    plot_path = os.path.join(plot_dir, 'loss_convergence_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"📊 그래프 저장 완료! 확인해 보세요 ➔ {plot_path}")

if __name__ == '__main__':
    main()