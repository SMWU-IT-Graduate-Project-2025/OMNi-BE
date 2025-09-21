import onnxruntime as ort
import os
import torch
import numpy as np
from typing import Optional

class ONNXVisionModel:
    """
    ONNX 비전 모델을 사용한 이미지 임베딩 추출 클래스
    """

    def __init__(self, model_path: str, providers: Optional[list] = None):
        """
        ONNX 비전 모델 초기화

        Args:
            model_path: ONNX 모델 파일 경로
            providers: ONNX 런타임 프로바이더 목록
        """
        self.model_path = model_path

        # 기본 프로바이더 설정
        if providers is None:
            try:
                # CoreML 우선 (Apple Silicon)
                providers = [
                    (
                        # "CoreMLExecutionProvider", CoreML이 입력 차원 16384를 초과하는 텐서(tensor)를 지원하지 못함.
                        # {
                        #     "ModelFormat": "MLProgram",
                        #     "MLComputeUnits": "ALL",
                        #     "RequireStaticInputShapes": "0",
                        #     "EnableOnSubgraphs": "0",
                        # },
                    ),
                    "CPUExecutionProvider",
                ]
            except Exception as e:
                print(f"Warning: CoreML provider unavailable: {e}, falling back to CPU")
                providers = ["CPUExecutionProvider"]

        # ONNX 세션 생성
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 입력/출력 정보 확인
        self.inputs = {inp.name: inp for inp in self.session.get_inputs()}
        self.outputs = {out.name: out for out in self.session.get_outputs()}

        print(f"ONNX Vision Model loaded from: {model_path}")
        print(f"Using providers: {self.session.get_providers()}")
        print(f"Inputs: {list(self.inputs.keys())}")
        print(f"Outputs: {list(self.outputs.keys())}")

    def get_image_embedding(self, image_tensor: torch.Tensor, text: str = "dummy text") -> torch.Tensor:
        """
        이미지 텐서에서 임베딩을 추출
        
        Args:
            image_tensor: 이미지 텐서
            text: 더미 텍스트 (모델이 텍스트 입력을 요구하므로)
        """
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.cpu().numpy().astype(np.float32)
        else:
            image_np = image_tensor.astype(np.float32)

        # 더미 텍스트 입력 생성 (최소한의 토큰)
        batch_size = image_np.shape[0]
        sequence_length = 1  # 최소 길이
        
        # 더미 input_ids와 attention_mask 생성
        input_ids = np.array([[49406, 49407]], dtype=np.int64)  # CLIP의 시작/끝 토큰
        attention_mask = np.array([[1, 1]], dtype=np.int64)
        
        # 배치 크기에 맞게 확장
        if batch_size > 1:
            input_ids = np.tile(input_ids, (batch_size, 1))
            attention_mask = np.tile(attention_mask, (batch_size, 1))

        # 입력 구성
        input_dict = {
            "pixel_values": image_np,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # 추론 실행
        outputs = self.session.run(["image_embeds"], input_dict)
        image_embedding = torch.from_numpy(outputs[0])

        # 정규화
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding


def load_onnx_vision_model(model_path: str) -> ONNXVisionModel:
    """
    ONNX 비전 모델을 로드하는 헬퍼 함수
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")

    return ONNXVisionModel(model_path)
