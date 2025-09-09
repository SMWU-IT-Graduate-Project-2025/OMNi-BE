import pandas as pd
import numpy as np
import torch
from typing import Dict, Optional
import os

class EventEmbeddingLoader:
    """데모 이벤트 임베딩을 로드하고 관리하는 클래스"""
    
    def __init__(self, csv_path: str = "public/demo_event_embeddings.csv"):
        """
        Args:
            csv_path: 데모 이벤트 임베딩 CSV 파일 경로
        """
        self.csv_path = csv_path
        self.event_embeddings: Dict[str, torch.Tensor] = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """CSV 파일에서 이벤트 임베딩을 로드"""
        try:
            # CSV 파일 읽기
            df = pd.read_csv(self.csv_path)
            
            # 각 행(이벤트)에 대해 임베딩 추출
            for _, row in df.iterrows():
                event_label = row['event_label']
                
                # 임베딩 컬럼들 추출 (emb_0부터 emb_511까지)
                embedding_cols = [col for col in df.columns if col.startswith('emb_')]
                embedding_values = row[embedding_cols].values.astype(np.float32)
                
                # numpy array를 torch tensor로 변환
                embedding_tensor = torch.tensor(embedding_values, dtype=torch.float32)
                
                # 정규화
                embedding_tensor = embedding_tensor / embedding_tensor.norm(dim=-1, keepdim=True)
                
                self.event_embeddings[event_label] = embedding_tensor
                
            print(f"Loaded {len(self.event_embeddings)} event embeddings from {self.csv_path}")
            
        except Exception as e:
            print(f"Error loading embeddings from {self.csv_path}: {e}")
            raise
    
    def get_embedding(self, event_label: str) -> Optional[torch.Tensor]:
        """
        특정 이벤트 라벨에 해당하는 임베딩을 반환
        
        Args:
            event_label: 이벤트 라벨 (예: "Normal", "V", "Hello", "Thumbs-up")
            
        Returns:
            해당 이벤트의 임베딩 텐서 또는 None
        """
        return self.event_embeddings.get(event_label)
    
    def get_available_events(self) -> list:
        """사용 가능한 이벤트 라벨 목록을 반환"""
        return list(self.event_embeddings.keys())
    
    def get_embedding_by_query_label(self, query_label: str) -> Optional[torch.Tensor]:
        """
        쿼리 라벨로 임베딩을 찾는 함수
        한국어 라벨을 영어 라벨로 매핑하여 임베딩을 반환
        
        Args:
            query_label: 한국어 이벤트 라벨 (예: "브이 하기", "안녕", "엄지척")
            
        Returns:
            해당 이벤트의 임베딩 텐서 또는 None
        """
        # 한국어 라벨을 영어 라벨로 매핑
        label_mapping = {
            "브이 하기": "V",
            "안녕": "Hello", 
            "엄지척": "Thumbs-up",
            "정상": "Normal"
        }
        
        english_label = label_mapping.get(query_label)
        if english_label:
            return self.get_embedding(english_label)
        
        # 직접 영어 라벨로 검색
        return self.get_embedding(query_label)

# 전역 인스턴스 생성
event_loader = EventEmbeddingLoader()
