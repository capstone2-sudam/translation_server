# scripts/find_max_angle.py

import os
import json
import numpy as np

def find_robust_max_angle(dataset_folder_path: str):
    """
    변환 완료된 센서 데이터셋(JSON)을 순회하며 실질적인 최대 굽힘 각도를 찾습니다.
    """
    all_flex_values = []
    
    print(f"🔍 '{dataset_folder_path}' 폴더 스캔 중...")
    
    # 1. 폴더 내의 모든 JSON 파일 순회
    for root, dirs, files in os.walk(dataset_folder_path):
        for file in files:
            # 우리가 변환한 센서 데이터 파일만 타겟으로 삼음
            if file.endswith("_converted_sensor.json"):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                     
                    # 2. 파일 내의 프레임 순회
                    for frame in data.get("frames", []):
                        
                        # 왼손 Flex 데이터 추출
                        left_flex = frame.get("left_hand", {}).get("flex_mcp_pip", [])
                        # 결측치([0.0, 0.0, ...])가 아닌 경우에만 수집
                        if left_flex and not np.allclose(left_flex, 0.0):
                            all_flex_values.extend(left_flex)
                            
                        # 오른손 Flex 데이터 추출
                        right_flex = frame.get("right_hand", {}).get("flex_mcp_pip", [])
                        if right_flex and not np.allclose(right_flex, 0.0):
                            all_flex_values.extend(right_flex)

    if not all_flex_values:
        print("⚠️ 에러: 처리된 관절 데이터가 없거나 결측치만 존재합니다.")
        return

    # 3. 분석 결과 계산 (🌟 수정된 부분)
    all_flex_values = np.array(all_flex_values)
    average_val = np.mean(all_flex_values)

    # 퍼센타일(백분위수) 구간별 값 추출
    max_100 = np.max(all_flex_values)
    p_99_9 = np.percentile(all_flex_values, 99.9)
    p_99 = np.percentile(all_flex_values, 99.0)
    p_98 = np.percentile(all_flex_values, 98.0)
    p_95 = np.percentile(all_flex_values, 95.0)

    # 4. 결과 출력 (🌟 수정된 부분)
    print("\n" + "="*35)
    print("📊 데이터셋 센서 각도 분석 결과")
    print("="*35)
    print(f"총 분석된 관절 센서 데이터: {len(all_flex_values):,} 개")
    print(f"평균 굽힘 각도             : {average_val:.2f} 도")
    
    print("\n[🔎 Flex 센서 분포 확인]")
    print(f"  - 100.0% (절대 최대치) : {max_100:.2f} 도")
    print(f"  -  99.9% (극단치 제외) : {p_99_9:.2f} 도")
    print(f"  -  99.0% (일반적 기준) : {p_99:.2f} 도")
    print(f"  -  98.0% (안정적 기준) : {p_98:.2f} 도")
    print(f"  -  95.0% (공격적 기준) : {p_95:.2f} 도")
    print("="*35)
    print("💡 가이드: 출력된 분포를 보고 숫자가 급격히 떨어지다가")
    print("   완만해지며 안정화되는 지점(통상 98~99%)의 값을")
    print("   모델 학습 시 정규화 상수로 선택하세요!")

# 스크립트 실행
if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1단계 파이프라인에서 센서 데이터를 저장했던 폴더 경로를 지정하세요.
    DATASET_PATH = os.path.join(CURRENT_DIR, "dataset", "translation_dataset", "converted_sensor")
    
    # 경로가 존재하는지 확인 후 실행
    if os.path.exists(DATASET_PATH):
        find_robust_max_angle(DATASET_PATH)
    else:
        print(f"❌ 경로를 찾을 수 없습니다: {DATASET_PATH}")



'''
실행 결과

🔍 'c:\Users\444vip\Sudam\dataset_processing\dataset\translation_dataset\converted_sensor' 폴더 스캔 중...
===================================
📊 데이터셋 센서 각도 분석 결과
===================================
총 분석된 관절 센서 데이터: 175,565,560 개
평균 굽힘 각도             : 31.90 도

[🔎 Flex 센서 분포 확인]
  - 100.0% (절대 최대치) : 179.96 도
  -  99.9% (극단치 제외) : 168.28 도
  -  99.0% (일반적 기준) : 146.18 도
  -  98.0% (안정적 기준) : 134.10 도
  -  95.0% (공격적 기준) : 106.34 도
===================================
💡 가이드: 출력된 분포를 보고 숫자가 급격히 떨어지다가
   완만해지며 안정화되는 지점(통상 98~99%)의 값을
   모델 학습 시 정규화 상수로 선택하세요!
'''