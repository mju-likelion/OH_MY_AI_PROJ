# 10/6 세미나 — Tabular Foundation Model과 기존 머신러닝의 조건별 비교

**연구 질문**: 데이터 크기, 결측률, 범주형 비율, 클래스 불균형에 따라 사전학습 모델(NVIDIA Kumo Tabular, TabICLv2)과 부스팅 모델(XGBoost·LightGBM·CatBoost)의 우위가 어떻게 달라지는가?

산출물은 "새 모델이 더 좋다"가 아니라 **어떤 데이터에서 어떤 모델을 선택해야 하는가**(결정 가이드)입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `tfm_vs_gbdt_benchmark.ipynb` | 실험 노트북 (Colab GPU용). 설치 → 데이터 → 조건 변환 → 모델 → 실행 → 분석 → 결정 가이드 |
| `results_local/` | 로컬 Mac CPU에서 `quick` 모드로 돌린 예비 결과 (Kumo **small**, seed 1개) |

## 실행 방법 (Colab)

1. 노트북을 Colab에 업로드 → 런타임 → 런타임 유형 변경 → **GPU** 선택
2. 설정 셀에서 `MODE`를 고릅니다
   - `"quick"`: 세미나 데모용 (데이터셋 6개 + 통제 실험 2개, seed 1개, 튜닝 10회)
   - `"full"`: 결론용 (데이터셋 12개 + 통제 실험 4개, seed 3개, 튜닝 30회). 몇 시간이 걸립니다
3. 세션이 끊길 수 있다면 `USE_DRIVE = True`로 바꿔 Google Drive에 저장하세요. 같은 셀을 다시 실행하면 끝난 조합은 건너뛰고 이어서 진행합니다.
4. 결과: `results/results_<MODE>.csv`, `results/decision_guide_<MODE>.csv`, `results/figures/*.png`

## 로컬(macOS)에서 실행할 때

macOS에서는 PyTorch와 XGBoost/LightGBM이 서로 다른 OpenMP 런타임을 불러와 커널이 죽습니다. 단일 스레드로 실행하세요(느립니다).

```bash
OMP_NUM_THREADS=1 TFM_N_JOBS=1 jupyter lab
```

## 참고 자료

- NVIDIA, [Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction](https://huggingface.co/blog/nvidia/kumo-tabular), 2026.09.29
- [Kumo-Tabular 모델 카드](https://huggingface.co/nvidia/Kumo-Tabular) · [structured-data-models (sdm)](https://github.com/NVIDIA/structured-data-models)
