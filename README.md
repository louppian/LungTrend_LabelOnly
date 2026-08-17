# CXR_GUI_V5

흉부 X-ray 중증도 평가 GUI (Gradio + DORGA / GUNet / STN).

## 생짜 PC 설치 순서 (오프라인 배포)

### 사전 준비 — 온라인 PC에서

1. `download_packages.bat` 실행
   - `offline_packages/` — 일반 패키지 휠 (CPU torch 휠은 자동 제거됨)
   - `offline_packages_torch/` — CUDA torch 휠 (약 2.7GB)
2. 프로젝트 폴더 **전체**를 USB 등으로 대상 PC에 복사
   - ⚠️ `assets/weights/` 가중치 3개는 git에 없음 — 반드시 포함할 것:
     `DORGAFull_posthoc.pth`, `finetuned_9601.pt`, `stn_weights.pth`

### 대상 PC에서 (순서대로)

| 순서 | 할 일 | 비고 |
|---|---|---|
| 1 | NVIDIA 드라이버 설치 | CUDA 11.8 호환 (최신 드라이버면 됨) |
| 2 | Anaconda/Miniconda 설치 | |
| 3 | `conda create -n torch_ev python=3.10 -y` | Anaconda Prompt에서 |
| 4 | `install_offline.bat` 실행 | CUDA torch → 나머지 패키지 순서로 설치됨 |
| 5 | 설치 로그 끝의 `CUDA: True` 확인 | `False`면 torch가 CPU 빌드 — 중단하고 확인 |
| 6 | `run.bat` 실행 | 브라우저 자동 오픈 (7860, 사용 중이면 7861~) |

### 온라인 PC에 바로 설치할 때

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## 파일 구성

- `requirements.txt` — torch(CUDA 핀) + `requirements_extra.txt` 포함 (전체 설치용)
- `requirements_extra.txt` — torch 제외 나머지 패키지 (오프라인 배포용 실제 목록)
- `run.bat` — 앱 실행 (7860 점유 프로세스 정리 후 시작)
- `download_packages.bat` / `install_offline.bat` — 오프라인 배포용
- `assets/weights/` — 모델 가중치 (git 미포함)
- `results/` — export 결과 (NPZ·이미지·`roi_export.xlsx`)

## 주의

- **torch_ev 환경에 다른 프로젝트 패키지를 설치하지 말 것** — CPU 빌드 torch가
  CUDA 빌드를 덮어써서 추론이 수 배 느려지는 사고가 실제로 있었음 (2026-07-16).
  다른 프로젝트는 별도 conda 환경 사용.
