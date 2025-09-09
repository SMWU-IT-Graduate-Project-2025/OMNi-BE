from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import torch
import numpy as np
from typing import Tuple
import io

def preprocess_image(image_bytes: bytes, size: int = 224) -> torch.Tensor:
    """
    이미지 바이트를 전처리하여 텐서로 변환
    
    Args:
        image_bytes: 이미지 바이트 데이터
        size: 이미지 크기 (기본값: 224)
        
    Returns:
        전처리된 이미지 텐서 [1, 3, size, size]
    """
    # PIL 이미지로 변환
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 전처리 파이프라인
    transform = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 텐서로 변환하고 배치 차원 추가
    image_tensor = transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]
    
    return image_tensor

def get_image_embedding(image_tensor: torch.Tensor, vision_model) -> torch.Tensor:
    """
    이미지 텐서에서 임베딩을 추출
    
    Args:
        image_tensor: 전처리된 이미지 텐서 [1, 3, 224, 224]
        vision_model: CLIP 비전 모델
        
    Returns:
        이미지 임베딩 텐서 [1, embedding_dim]
    """
    # 모델의 디바이스로 이동 (CUDA, MPS, 또는 CPU)
    device = next(vision_model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 이미지 임베딩 추출
        image_embedding = vision_model(pixel_values=image_tensor)["image_embeds"]
        
        # 정규화
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    return image_embedding

def get_text_embedding(text: str, text_model, tokenizer) -> torch.Tensor:
    """
    텍스트에서 임베딩을 추출
    
    Args:
        text: 입력 텍스트
        text_model: CLIP 텍스트 모델
        tokenizer: CLIP 토크나이저
        
    Returns:
        텍스트 임베딩 텐서 [1, embedding_dim]
    """
    # 모델의 디바이스 확인
    device = next(text_model.parameters()).device
    
    # 텍스트 토크나이징
    inputs = tokenizer(text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 텍스트 임베딩 추출
        outputs = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_embedding = outputs[0]
        
        # 정규화
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    return text_embedding

def calculate_similarity(image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
    """
    이미지와 텍스트 임베딩 간의 코사인 유사도 계산
    
    Args:
        image_embedding: 이미지 임베딩 [1, embedding_dim]
        text_embedding: 텍스트 임베딩 [1, embedding_dim]
        
    Returns:
        코사인 유사도 점수 (0~1)
    """
    # 코사인 유사도 계산
    similarity = torch.matmul(image_embedding, text_embedding.T).squeeze()
    
    # CPU로 이동하고 float으로 변환
    similarity_score = similarity.cpu().item()
    
    return similarity_score

def calculate_image_demo_similarity(image_embedding: torch.Tensor, demo_embedding: torch.Tensor) -> float:
    """
    이미지 임베딩과 데모 이벤트 임베딩 간의 코사인 유사도 계산
    
    Args:
        image_embedding: 이미지 임베딩 [1, embedding_dim]
        demo_embedding: 데모 이벤트 임베딩 [embedding_dim]
        
    Returns:
        코사인 유사도 점수 (0~1)
    """
    # 데모 임베딩을 배치 차원 추가하여 이미지 임베딩과 같은 형태로 만들기
    demo_embedding_batch = demo_embedding.unsqueeze(0)  # [1, embedding_dim]
    
    # GPU로 이동
    demo_embedding_batch = demo_embedding_batch.to(image_embedding.device)
    
    # 코사인 유사도 계산
    similarity = torch.matmul(image_embedding, demo_embedding_batch.T).squeeze()
    
    # CPU로 이동하고 float으로 변환
    similarity_score = similarity.cpu().item()
    
    return similarity_score
