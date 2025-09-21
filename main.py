import psycopg2
from dotenv import load_dotenv
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection
import torch
from core.event_loader import event_loader
from core.image_utils import (
    preprocess_image, 
    get_image_embedding, 
    calculate_image_demo_similarity
)
from typing import Dict, Optional

# 프레임 간 유사도 변화 추적을 위한 전역 변수
previous_similarities: Dict[str, float] = {}  # {query_label: previous_similarity}
frame_count: Dict[str, int] = {}  # {query_label: frame_count}
event_active: Dict[str, bool] = {}

# init models
t_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
v_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
v_model = v_model.eval()

# GPU 사용 가능한 경우 GPU로 이동 (CUDA 또는 MPS)
if torch.cuda.is_available():
    t_model = t_model.cuda()
    v_model = v_model.cuda()
    print("Models loaded on CUDA GPU")
elif torch.backends.mps.is_available():
    t_model = t_model.to('mps')
    v_model = v_model.to('mps')
    print("Models loaded on MPS (Apple Silicon GPU)")
else:
    print("Models loaded on CPU")

print("Models initialized successfully")

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="OMNi VLM Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 멀티클래스 이벤트 감지 결과 모델
class EventDetectionResult(BaseModel):
    event_label: str
    similarity_score: float
    detected: bool
    active: bool

# 응답 모델 정의
class InferenceResponse(BaseModel):
    success: bool
    demo_event: list[EventDetectionResult]  # 데모 이벤트들 (사용자 선택 이벤트 포함)
    store_abnormal: list[EventDetectionResult]  # 저장된 비정상 이벤트들 (Fall, Arson 등)
    message: str
    threshold: float  # 기본 임계값

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Example query
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    print(f"Failed to connect: {e}")



@app.post("/api/vlm/inference", response_model=InferenceResponse)
async def vlm_inference(
    image: UploadFile = File(..., description="웹캠 캡처 이미지 (JPEG)"),
    query: str = Form(..., description="사용자가 선택한 이벤트의 영어 설명"),
    query_label: str = Form(..., description="사용자가 선택한 이벤트의 한국어 라벨")
):
    """
    VLM 모델을 활용한 실시간 이벤트 감지
    
    Args:
        image: 웹캠 캡처 이미지 (JPEG Blob)
        query: 사용자가 선택한 이벤트의 영어 설명 (예: "Someone is making a V-shape with two fingers")
        query_label: 사용자가 선택한 이벤트의 한국어 라벨 (예: "브이 하기")
    
    Returns:
        InferenceResponse: 이벤트 감지 결과
    """
    try:
        # 이미지 파일 검증
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 이미지 바이트 읽기
        image_bytes = await image.read()
        
        # 이미지 전처리
        image_tensor = preprocess_image(image_bytes)
        
        # 이미지 임베딩 추출
        image_embedding = get_image_embedding(image_tensor, v_model)
        
        # 데모 이벤트와 저장된 비정상 이벤트를 분리하여 계산
        demo_event_results = []
        store_abnormal_event_results = []
        primary_similarity = 0.0
        
        # 사용자가 선택한 이벤트의 유사도 계산
        demo_embedding = event_loader.get_embedding_by_query_label(query_label)
        if demo_embedding is not None:
            primary_similarity = calculate_image_demo_similarity(image_embedding, demo_embedding)
        else:
            logger.warning(f"Demo embedding not found for query_label: {query_label}")
            # 데모 임베딩이 없는 경우 텍스트 임베딩 생성
            try:
                # 텍스트 토큰화 및 임베딩 추출
                text_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
                
                # GPU 사용 가능한 경우 텐서를 GPU로 이동
                if torch.cuda.is_available():
                    text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
                elif torch.backends.mps.is_available():
                    text_inputs = {k: v.to('mps') for k, v in text_inputs.items()}
                
                # 텍스트 임베딩 추출
                with torch.no_grad():
                    text_outputs = t_model(**text_inputs)
                    text_embedding = text_outputs.text_embeds
                    
                    # 정규화
                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                
                # 이미지와 텍스트 임베딩 간 유사도 계산
                primary_similarity = torch.cosine_similarity(image_embedding, text_embedding, dim=-1).item()
                logger.info(f"Generated text embedding for query: {query}, similarity: {primary_similarity:.4f}")
                
            except Exception as e:
                logger.error(f"Error generating text embedding: {str(e)}")
                primary_similarity = 0.0
        
        # 사용자가 선택한 이벤트를 데모 이벤트에 추가
        demo_event_results.append({
            'event_label': query_label,
            'similarity_score': primary_similarity,
            'detected': False,  # 나중에 설정
            'active': False     # 나중에 설정
        })
        
        # store_abnormal_event_embeddings.csv에서 Fall 이벤트만 유사도 계산
        store_events = event_loader.get_store_events()
        if 'Fall' in store_events:
            fall_embedding = store_events['Fall']
            similarity = calculate_image_demo_similarity(image_embedding, fall_embedding)
            store_abnormal_event_results.append({
                'event_label': 'Fall',
                'similarity_score': similarity,
                'detected': False,  # 나중에 설정
                'active': False     # 나중에 설정
            })
        
        current_similarity = primary_similarity
        
        # 프레임 카운트 증가
        if query_label not in frame_count:
            frame_count[query_label] = 0
        frame_count[query_label] += 1
        
        # 멀티클래스 알람 판단 로직
        
        # 임계값 정의 (이벤트별로 다른 임계값 적용)
        if query_label == "카메라에 인사하기":
            similarity_threshold = 0.19  # 인사하기는 더 낮은 임계값
        else:
            similarity_threshold = 0.2   # 기본 임계값
        
        gap_threshold = 0.05         # 순간 변화 판단 (새 이벤트 트리거)
        
        # 데모 이벤트에 대해 감지 상태 판단
        for event_result in demo_event_results:
            event_label = event_result['event_label']
            similarity = event_result['similarity_score']
            
            # 프레임 카운트 초기화 (각 이벤트별로)
            if event_label not in frame_count:
                frame_count[event_label] = 0
            frame_count[event_label] += 1
            
            # 이전 유사도 가져오기
            previous_similarity = previous_similarities.get(event_label, 0.0)
            similarity_gap = similarity - previous_similarity
            
            # 첫 프레임은 판단 스킵
            if frame_count[event_label] == 1:
                logger.info(f"First frame for {event_label}: Skipping alert")
                event_result['detected'] = False
                event_result['active'] = event_active.get(event_label, False)
            else:
                # 1. gap과 similarity 모두 만족 → 새 이벤트 발생
                if (similarity_gap >= gap_threshold 
                    and similarity >= similarity_threshold 
                    and not event_active.get(event_label, False)):   # 아직 활성화 안된 경우만
                    event_result['detected'] = True
                    event_result['active'] = True
                    event_active[event_label] = True
                    logger.info(f"New demo event triggered for {event_label}")

                # 2. 이미 발생한 이벤트 유지 중
                elif event_active.get(event_label, False):
                    if similarity >= similarity_threshold:
                        event_result['detected'] = False  # 유지 중엔 새 감지 아님
                        event_result['active'] = True
                        logger.info(f"Demo event still active for {event_label}")
                    else:
                        event_result['detected'] = False
                        event_result['active'] = False
                        event_active[event_label] = False
                        logger.info(f"Demo event ended for {event_label}")
                else:
                    event_result['detected'] = False
                    event_result['active'] = False
            
            # 현재 유사도를 다음 프레임을 위해 저장
            previous_similarities[event_label] = similarity
        
        # 저장된 비정상 이벤트에 대해 감지 상태 판단
        for event_result in store_abnormal_event_results:
            event_label = event_result['event_label']
            similarity = event_result['similarity_score']
            
            # 프레임 카운트 초기화 (각 이벤트별로)
            if event_label not in frame_count:
                frame_count[event_label] = 0
            frame_count[event_label] += 1
            
            # 이전 유사도 가져오기
            previous_similarity = previous_similarities.get(event_label, 0.0)
            similarity_gap = similarity - previous_similarity
            
            # 첫 프레임은 판단 스킵
            if frame_count[event_label] == 1:
                logger.info(f"First frame for {event_label}: Skipping alert")
                event_result['detected'] = False
                event_result['active'] = event_active.get(event_label, False)
            else:
                # 1. gap과 similarity 모두 만족 → 새 이벤트 발생
                if (similarity_gap >= gap_threshold 
                    and similarity >= similarity_threshold 
                    and not event_active.get(event_label, False)):   # 아직 활성화 안된 경우만
                    event_result['detected'] = True
                    event_result['active'] = True
                    event_active[event_label] = True
                    logger.info(f"New store abnormal event triggered for {event_label}")

                # 2. 이미 발생한 이벤트 유지 중
                elif event_active.get(event_label, False):
                    if similarity >= similarity_threshold:
                        event_result['detected'] = False  # 유지 중엔 새 감지 아님
                        event_result['active'] = True
                        logger.info(f"Store abnormal event still active for {event_label}")
                    else:
                        event_result['detected'] = False
                        event_result['active'] = False
                        event_active[event_label] = False
                        logger.info(f"Store abnormal event ended for {event_label}")
                else:
                    event_result['detected'] = False
                    event_result['active'] = False
            
            # 현재 유사도를 다음 프레임을 위해 저장
            previous_similarities[event_label] = similarity
        
        # 로깅
        logger.info(f"Query: {query}, Label: {query_label}")
        logger.info(f"Primary event similarity: {current_similarity:.4f}")
        
        # 감지된 이벤트들 로깅
        detected_demo_events = [r for r in demo_event_results if r['detected']]
        detected_store_abnormal_events = [r for r in store_abnormal_event_results if r['detected']]
        active_demo_events = [r for r in demo_event_results if r['active']]
        active_store_abnormal_events = [r for r in store_abnormal_event_results if r['active']]
        
        logger.info(f"Detected demo events: {[r['event_label'] for r in detected_demo_events]}")
        logger.info(f"Detected store abnormal events: {[r['event_label'] for r in detected_store_abnormal_events]}")
        logger.info(f"Active demo events: {[r['event_label'] for r in active_demo_events]}")
        logger.info(f"Active store abnormal events: {[r['event_label'] for r in active_store_abnormal_events]}")
        
        # 메시지 생성
        all_detected_events = detected_demo_events + detected_store_abnormal_events
        all_active_events = active_demo_events + active_store_abnormal_events
        
        if all_detected_events:
            message = f"새 이벤트 감지: {', '.join([r['event_label'] for r in all_detected_events])}"
        elif all_active_events:
            message = f"이벤트 유지 중: {', '.join([r['event_label'] for r in all_active_events])}"
        else:
            message = "이벤트 감지되지 않음"
        
        # EventDetectionResult 객체들로 변환
        demo_events = [EventDetectionResult(**result) for result in demo_event_results]
        store_abnormal_events = [EventDetectionResult(**result) for result in store_abnormal_event_results]
        
        return InferenceResponse(
            success=True,
            demo_event=demo_events,
            store_abnormal=store_abnormal_events,
            message=message,
            threshold=similarity_threshold
        )

    except Exception as e:
        logger.error(f"Error in VLM inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이벤트 감지 중 오류가 발생했습니다: {str(e)}")

@app.get("/api/vlm/events")
async def get_available_events():
    """사용 가능한 이벤트 목록을 반환"""
    try:
        events = event_loader.get_available_events()
        return {
            "success": True,
            "events": events,
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Error getting available events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이벤트 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/vlm/reset")
async def reset_frame_state():
    """프레임 상태 초기화 (새로운 세션 시작 시 사용)"""
    global previous_similarities, frame_count, event_active
    previous_similarities.clear()
    frame_count.clear()
    event_active.clear()
    
    logger.info("Frame state reset - starting new session")
    return {
        "success": True,
        "message": "Frame state reset successfully"
    }

@app.get("/api/vlm/health")
async def health_check():
    """서버 상태 확인"""
    # GPU 상태 확인
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_type": "unknown"
    }
    
    if torch.cuda.is_available():
        gpu_info["device_type"] = "cuda"
        gpu_info["cuda_device_count"] = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        gpu_info["device_type"] = "mps"
    else:
        gpu_info["device_type"] = "cpu"
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "gpu_info": gpu_info,
        "available_events": len(event_loader.get_available_events()),
        "store_events": len(event_loader.get_store_events()),
        "total_events": len(event_loader.get_all_events()),
        "active_sessions": len(frame_count),
        "active_events": len([k for k, v in event_active.items() if v])
    } 

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

        