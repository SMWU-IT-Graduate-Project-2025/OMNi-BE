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

# 응답 모델 정의
class InferenceResponse(BaseModel):
    success: bool
    event_detected: bool
    similarity_score: float
    query_label: str
    query_text: str
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
        
        # 데모 이벤트 임베딩과의 유사도 계산
        demo_embedding = event_loader.get_embedding_by_query_label(query_label)
        demo_similarity = 0.0
        
        if demo_embedding is not None:
            demo_similarity = calculate_image_demo_similarity(image_embedding, demo_embedding)
        else:
            logger.warning(f"Demo embedding not found for query_label: {query_label}")
        
        current_similarity = demo_similarity
        
        # 프레임 카운트 증가
        if query_label not in frame_count:
            frame_count[query_label] = 0
        frame_count[query_label] += 1
        
        # 알람 판단 로직
        alert = False
        
        if frame_count[query_label] == 1:
            # 첫 번째 프레임: 판단 스킵
            alert = False
            logger.info(f"First frame for {query_label}: Skipping alert")
        else:
            # 이전 프레임과 비교
            if query_label in previous_similarities:
                previous_similarity = previous_similarities[query_label]
                similarity_gap = current_similarity - previous_similarity
                
                # 유사도 증가 임계값 (gap threshold)
                gap_threshold = 0.05  # 이전 프레임 대비 gap 이상 증가해야 알람
                
                if similarity_gap >= gap_threshold:
                    alert = True
                    logger.info(f"Alert triggered: Gap {similarity_gap:.4f} >= {gap_threshold}")
                else:
                    alert = False
                    logger.info(f"No alert: Gap {similarity_gap:.4f} < {gap_threshold}")
        
        # 현재 유사도를 다음 프레임을 위해 저장
        previous_similarities[query_label] = current_similarity
        
        # 로깅
        logger.info(f"Query: {query}, Label: {query_label}")
        logger.info(f"Frame: {frame_count[query_label]}, Similarity: {current_similarity:.4f}, Alert: {alert}")
        
        # return InferenceResponse(
        #     similarity=current_similarity,
        #     event=query_label,
        #     alert=alert
        # )
        return InferenceResponse(
            success=True,
            event_detected=alert, # bool
            similarity_score=current_similarity,
            query_label=query_label,
            query_text=query,
            message=f"이벤트 {'감지됨' if alert else '감지되지 않음'}",
            threshold=gap_threshold
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
    global previous_similarities, frame_count
    previous_similarities.clear()
    frame_count.clear()
    
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
        "active_sessions": len(frame_count)
    } 

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

        