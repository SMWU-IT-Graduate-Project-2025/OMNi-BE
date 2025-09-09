#!/usr/bin/env python3
"""
OMNi VLM Inference API 테스트 스크립트

이 스크립트는 구현된 FastAPI 서버를 테스트하기 위한 예제입니다.
"""

import requests
import json
from pathlib import Path

# API 서버 URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """서버 상태 확인"""
    print("=== 서버 상태 확인 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/vlm/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data['status']}")
            print(f"✅ 모델 로드됨: {data['models_loaded']}")
            
            # GPU 정보 표시
            gpu_info = data.get('gpu_info', {})
            device_type = gpu_info.get('device_type', 'unknown')
            print(f"✅ 디바이스 타입: {device_type}")
            
            if device_type == 'cuda':
                print(f"✅ CUDA 사용 가능: {gpu_info.get('cuda_available', False)}")
                print(f"✅ CUDA 디바이스 수: {gpu_info.get('cuda_device_count', 0)}")
            elif device_type == 'mps':
                print(f"✅ MPS 사용 가능: {gpu_info.get('mps_available', False)}")
            else:
                print("ℹ️  CPU 모드로 실행 중")
            
            print(f"✅ 사용 가능한 이벤트 수: {data['available_events']}")
            return True
        else:
            print(f"❌ 서버 상태 확인 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return False

def test_get_events():
    """사용 가능한 이벤트 목록 조회"""
    print("\n=== 사용 가능한 이벤트 목록 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/vlm/events")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 이벤트 수: {data['count']}")
            print("📋 이벤트 목록:")
            for event in data['events']:
                print(f"  - {event}")
            return data['events']
        else:
            print(f"❌ 이벤트 목록 조회 실패: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 이벤트 목록 조회 실패: {e}")
        return []

def test_reset():
    """프레임 상태 초기화 테스트"""
    print("\n=== 프레임 상태 초기화 테스트 ===")
    try:
        response = requests.post(f"{BASE_URL}/api/vlm/reset")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 초기화 성공: {data['success']}")
            print(f"✅ 메시지: {data['message']}")
            return True
        else:
            print(f"❌ 초기화 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 초기화 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 OMNi VLM Inference API 테스트 시작")
    
    # 1. 서버 상태 확인
    if not test_health_check():
        print("❌ 서버가 실행되지 않았습니다. 서버를 먼저 시작해주세요.")
        return
    
    # 2. 사용 가능한 이벤트 목록 조회
    events = test_get_events()
    if not events:
        print("❌ 이벤트 목록을 가져올 수 없습니다.")
        return
    
    # 3. 프레임 상태 초기화 테스트
    test_reset()
    
    # 3. 테스트 이미지가 있다면 이벤트 감지 테스트
    test_image_path = "test_image.jpg"  # 테스트용 이미지 경로
    if Path(test_image_path).exists():
        # 예제 쿼리들
        test_cases = [
            {
                "query": "Someone is making a V-shape with two fingers",
                "query_label": "브이 하기"
            },
            {
                "query": "A person is waving his five fingers",
                "query_label": "안녕"
            },
            {
                "query": "A person is giving a thumbs-up sign",
                "query_label": "엄지척"
            },
            {
                "query": "Usual store conditions",
                "query_label": "정상"
            }
        ]
        
        for test_case in test_cases:
            test_inference(
                test_image_path,
                test_case["query"],
                test_case["query_label"]
            )
    else:
        print(f"⚠️  테스트 이미지가 없습니다: {test_image_path}")
        print("테스트 이미지를 준비하고 다시 실행해주세요.")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    main()
