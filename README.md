# OMNi-BE

OMNi VLM(Vision-Language Model) 기반 실시간 이벤트 감지 백엔드 서버

## 주요 기능

- **실시간 이벤트 감지**: 웹캠 이미지와 텍스트 쿼리를 기반으로 이벤트 발생 여부를 실시간으로 판단
- **VLM 모델 활용**: CLIP 모델을 사용한 이미지-텍스트 임베딩 유사도 계산
- **데모 이벤트 지원**: 미리 학습된 데모 이벤트 임베딩과의 유사도 비교
- **RESTful API**: FastAPI 기반의 고성능 웹 API

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`env.example` 파일을 참고하여 `.env` 파일을 생성하세요:

```bash
cp env.example .env
```

`.env` 파일에 다음 정보를 입력하세요:

```env
# 데이터베이스 설정 (PostgreSQL)
user=your_db_user
password=your_db_password
host=your_db_host
port=your_db_port
dbname=your_db_name
```

### 3. 모델 다운로드

서버 시작 시 CLIP 모델이 자동으로 다운로드됩니다:
- `Searchium-ai/clip4clip-webvid150k` (텍스트 모델)
- `Searchium-ai/clip4clip-webvid150k` (비전 모델)

## 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 시작되면 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 1. 이벤트 감지 (메인 기능)

**POST** `/api/vlm/inference`

웹캠 이미지와 쿼리를 받아 이벤트 감지 결과를 반환합니다.

**요청 형식**: `multipart/form-data`

**요청 파라미터**:
- `image`: 웹캠 캡처 이미지 (JPEG Blob)
- `query`: 사용자가 선택한 이벤트의 영어 설명 (예: "Someone is making a V-shape with two fingers")
- `query_label`: 사용자가 선택한 이벤트의 한국어 라벨 (예: "브이 하기")

**응답 예시**:
```json
{
  "similarity": 0.7564,
  "event": "브이 하기",
  "alert": true
}
```

### 2. 사용 가능한 이벤트 목록

**GET** `/api/vlm/events`

현재 로드된 데모 이벤트 목록을 반환합니다.

**응답 예시**:
```json
{
  "success": true,
  "events": ["Normal", "V", "Hello", "Thumbs-up"],
  "count": 4
}
```

### 3. 프레임 상태 초기화

**POST** `/api/vlm/reset`

새로운 세션을 시작할 때 프레임 상태를 초기화합니다.

**응답 예시**:
```json
{
  "success": true,
  "message": "Frame state reset successfully"
}
```

### 4. 서버 상태 확인

**GET** `/api/vlm/health`

서버 상태와 모델 로드 상태를 확인합니다.

**응답 예시**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "gpu_info": {
    "cuda_available": false,
    "mps_available": true,
    "device_type": "mps"
  },
  "available_events": 4,
  "active_sessions": 2
}
```

## 프로젝트 구조

```
OMNi-BE/
├── main.py                    # FastAPI 앱 메인 파일
├── core/
│   ├── event_loader.py        # 데모 이벤트 임베딩 로더
│   ├── image_utils.py         # 이미지 처리 및 임베딩 유틸리티
│   ├── inference_example.py   # 추론 예제 코드
│   └── utils.py              # 기타 유틸리티 함수
├── public/
│   └── demo_event_embeddings.csv  # 데모 이벤트 임베딩 데이터
├── test_api.py               # API 테스트 스크립트
├── requirements.txt          # Python 의존성
├── env.example              # 환경 변수 예시
└── README.md                # 프로젝트 문서
```

## 주요 특징

### 이벤트 감지 알고리즘

1. **이미지 전처리**: 웹캠 이미지를 CLIP 모델에 맞게 전처리
2. **이미지 임베딩 추출**: CLIP 비전 모델로 이미지 임베딩 생성
3. **데모 임베딩 비교**: 미리 학습된 데모 이벤트 임베딩과 비교
4. **유사도 계산**: 이미지 임베딩과 데모 이벤트 임베딩 간의 코사인 유사도 계산
5. **프레임 간 변화 감지**: 이전 프레임 대비 유사도 증가량이 임계값(gap) 이상일 때만 알람 활성화
6. **첫 프레임 스킵**: 최초 웹캠 이미지는 알람 판단에서 제외

### 지원하는 이벤트

현재 지원하는 이벤트들:
- **Normal**: 일반적인 상황
- **V**: 브이 하기 (두 손가락으로 V자 만들기)
- **Hello**: 안녕 (손 흔들기)
- **Thumbs-up**: 엄지척 (엄지손가락 올리기)

### 성능 최적화

- **GPU 가속**: CUDA 사용 가능 시 자동으로 GPU 활용
- **Apple Silicon 지원**: macOS에서 MPS(Metal Performance Shaders) 지원
- **모델 캐싱**: 서버 시작 시 모델을 메모리에 로드하여 추론 속도 향상
- **배치 처리**: 이미지 전처리 및 임베딩 추출 최적화

## 테스트

API 테스트를 위해 제공된 테스트 스크립트를 사용할 수 있습니다:

```bash
python test_api.py
```

테스트 스크립트는 다음 기능을 테스트합니다:
- 서버 상태 확인
- 사용 가능한 이벤트 목록 조회
- 이벤트 감지 API 호출

## 문제 해결

### 모델 로딩 오류
- 인터넷 연결 상태 확인
- GPU 메모리 부족 시 CPU 모드로 자동 전환

### 이미지 업로드 오류
- 지원되는 이미지 형식: JPEG, PNG, GIF 등
- 이미지 크기 제한 확인

### 성능 이슈
- GPU 사용 가능 여부 확인 (`/api/vlm/health` 엔드포인트)
- 모델 로드 상태 확인
- macOS에서 MPS 지원 확인

## 개발자 정보

이 프로젝트는 OMNi CCTV 시스템의 백엔드 서버로, VLM 모델을 활용한 실시간 이벤트 감지 기능을 제공합니다.