# scripts/merge_dataset.py

import os
import json
import numpy as np
from tqdm import tqdm


# 센서 데이터 정규화 함수
def normalize_sensor(hand_data, global_max_flex):
    if not hand_data:
        return {"flex_mcp_pip": [0.0]*10, "imu_quat": [0.0, 0.0, 0.0, 1.0]}
    
    raw_flex = np.array(hand_data.get('flex_mcp_pip', [0.0]*10))
    scaled_flex = np.clip(raw_flex / global_max_flex, 0.0, 1.0)
    
    raw_imu = np.array(hand_data.get("imu_quat", [0.0, 0.0, 0.0, 1.0]))

    if np.any(np.isnan(raw_imu)) or np.any(np.isinf(raw_imu)):
        raw_imu = np.array([0.0, 0.0, 0.0, 1.0])

    # sign fix (valid)
    if raw_imu[3] < 0:
        raw_imu = -raw_imu

    scaled_imu = raw_imu
    
    return {"flex_mcp_pip": scaled_flex.tolist(), "imu_quat": scaled_imu.tolist()}


def process_and_merge_datasets(vision_dir, sensor_dir, global_max_flex, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 98% 기준 안정값을 상수로 고정
    print(f"🔥 정규화 기준 각도 고정: {global_max_flex} 도")

    # ---------------------------------------------------------
    # Step 1: 비전(Vision) 파일과 센서(Sensor) 파일 탐색 및 딕셔너리 매핑
    # ---------------------------------------------------------
    print("🔍 하위 폴더 스캔 및 파일 짝 맞추기 진행 중...")

    vision_map = {}
    sensor_map = {}

    # 비전 데이터 탐색
    for root, _, files in os.walk(vision_dir):
        for file in files:
            if file.endswith('_keypoints_final.json'):
                # 파일명에서 고유 ID 추출 (예: VXPAKOKS230000010)
                base_id = file.replace('_keypoints_final.json', '')
                vision_map[base_id] = os.path.join(root, file)

    # 센서 데이터 탐색
    for root, _, files in os.walk(sensor_dir):
        for file in files:
            if file.endswith('_converted_sensor.json'):
                # 파일명에서 고유 ID 추출
                base_id = file.replace('_converted_sensor.json', '')
                sensor_map[base_id] = os.path.join(root, file)

    
    # 두 데이터셋에 공통으로 존재하는 ID만 추출 (교집합)
    common_ids = set(vision_map.keys()).intersection(set(sensor_map.keys()))
    print(f"✅ 총 {len(common_ids):,} 개의 매칭되는 데이터 쌍을 찾았습니다!\n")
    
    # ---------------------------------------------------------
    # Step 2: 매칭된 파일들을 열어서 정규화 및 병합
    # ---------------------------------------------------------
    for base_id in tqdm(common_ids):
        v_path = vision_map[base_id]
        s_path = sensor_map[base_id]

        with open(v_path, 'r', encoding='utf-8') as f:
            v_data = json.load(f)
        with open(s_path, 'r', encoding='utf-8') as f:
            s_data = json.load(f)

        # 프레임 번호를 기준으로 센서 데이터를 딕셔너리로 매핑
        sensor_frames = {f['frame_number']: f for f in s_data.get('frames', [])}
        
        merged_frames = []
        for v_frame in v_data.get('frames', []):
            f_num = v_frame['frame_number']
            s_frame = sensor_frames.get(f_num, {})
            
            # 비전 데이터에 정규화된 센서 데이터를 추가 (통합 딕셔너리 생성)
            merged_frame = v_frame.copy() 
            merged_frame['sensor_left_hand'] = normalize_sensor(s_frame.get('left_hand'), global_max_flex)
            merged_frame['sensor_right_hand'] = normalize_sensor(s_frame.get('right_hand'), global_max_flex)
                
            assert len(merged_frame['sensor_left_hand']["flex_mcp_pip"]) == 10, \
            f"{base_id} left flex={len(merged_frame['sensor_left_hand']['flex_mcp_pip'])}"

            assert len(merged_frame['sensor_left_hand']["imu_quat"]) == 4, \
                f"{base_id} left imu={len(merged_frame['sensor_left_hand']['imu_quat'])}"

            assert len(merged_frame['sensor_right_hand']["flex_mcp_pip"]) == 10, \
                f"{base_id} right flex={len(merged_frame['sensor_right_hand']['flex_mcp_pip'])}"

            assert len(merged_frame['sensor_right_hand']["imu_quat"]) == 4, \
                f"{base_id} right imu={len(merged_frame['sensor_right_hand']['imu_quat'])}"

            total_sensor_dim = (
                len(merged_frame['sensor_left_hand']["flex_mcp_pip"]) +
                len(merged_frame['sensor_left_hand']["imu_quat"]) +
                len(merged_frame['sensor_right_hand']["flex_mcp_pip"]) +
                len(merged_frame['sensor_right_hand']["imu_quat"])
            )

            assert total_sensor_dim == 28, \
                f"{base_id} total sensor dim={total_sensor_dim}"

            merged_frames.append(merged_frame)
            
        # 완성된 통합 데이터 저장
        merged_package = {
            "metadata": v_data.get("metadata", {}),
            "frames": merged_frames
        }
        
        filename = f"{base_id}_merged.json"

        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_package, f, ensure_ascii=False)


if __name__ == "__main__":
    global_max_flex = 135.0 
    video_dir = r"C:\Users\444vip\Sudam\translation_server\dataset\vision_dataset"
    sensor_dir = r"C:\Users\444vip\Sudam\translation_server\dataset\sensor_dataset"
    output_dir = r"C:\Users\444vip\Sudam\translation_server\dataset\merged_dataset"
    process_and_merge_datasets(video_dir, sensor_dir, global_max_flex, output_dir)
