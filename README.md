# 간결한 모델별 실행 코드

## 파일 구성

- `모델명_Test.py`: 해당 모델 하나만 3구간 시간순 검증하고 테스트합니다.
- `Best_Way.py`: 선정된 HGB만 학습·평가하며, 파일 끝에 최적화 코드를 바로 붙일 수 있습니다.
- 공용 코드 파일, 기존 프로젝트 import, 자동 보고서 생성은 없습니다. 각 파일은 독립적입니다.

11개 테스트 파일: Profile, Ridge, ElasticNet, KNN, SVR, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HGB, MLP.

## 실행

Python 3.12 환경에서:

```bash
python -m pip install -r requirements.txt
python HGB_Test.py
python Best_Way.py
```

`okm_augumented_2021.csv`를 코드와 같은 폴더에 두세요. 원본 데이터는 배포본에 포함하지 않았습니다. 다른 위치라면 파일 위쪽의 `DATA_PATH`를 수정하거나 다음처럼 지정합니다.

```bash
export POWER_DATA_PATH="/실제/경로/okm_augumented_2021.csv"
python Best_Way.py
```

각 파일은 요청한 모델만 실행하며 결과를 화면에 출력합니다. CSV·캐시·보고서를 추가 생성하지 않습니다.

## 최적화 코드 이어 붙이기

`Best_Way.py` 끝의 표시된 위치에 코드를 작성합니다. 그 시점에는 다음 변수가 준비돼 있습니다.

| 이름 | 내용 |
| --- | --- |
| `model` | 전처리를 포함해 학습된 HGB 파이프라인 |
| `df` | 정제된 전체 데이터, 시간 인덱스 |
| `train`, `valid`, `test` | 시간순으로 분리된 데이터 |
| `FEATURES`, `TARGETS` | 학습 입력 10개와 전력 출력 4개의 순서 |
| `predict_power(candidate)` | 후보 입력의 전력 4개를 예측하는 함수 |
| `test_prediction`, `test_metrics` | 테스트 예측과 MAE·RMSE·R² |

예시:

```python
candidate = test.iloc[:24].copy()
candidate["생산량"] *= 0.9
predicted_power = predict_power(candidate)
```

이는 함수 호출 예시일 뿐, 총생산량·설비 제약을 만족하는 최적화안이 아닙니다. 후보에는 `생산량`, `기온`, `풍속`, `습도`, `강수량` 열과 유효한 `DatetimeIndex`가 필요합니다. 함수가 시간 변수를 다시 만들기 때문에 날짜·시간을 바꾸어도 과거의 파생변수를 재사용하지 않습니다. 실제 전력 정답 열은 필요하지 않습니다. `model.predict()`를 직접 호출할 때는 `make_features(candidate)`로 입력을 만들고 음수 처리도 맞춰야 하므로 `predict_power()` 사용을 권장합니다.

## 이전 분석과의 관계

전처리, 모델별 설정, 시간순 검증 구간은 기존 분석과 같습니다. 테스트 학습 기간은 2021-01-01~06-27이며 검증·테스트 자료까지 재학습하지 않습니다. `Best_Way.py`는 선정이 끝난 HGB만 실행하므로 검증 3회를 반복하지 않습니다.

기존 다기준 선정은 정확도 60%, 안정성 25%, 학습·검증 차이 15%에서 HGB 1위였습니다. 이 간결한 파일 묶음은 모델 간 종합 순위를 다시 계산하지 않고 개별 검증 지표를 제공합니다. R²·실행 시간은 참고값입니다. 과소예측률·적중률·피크 지표 및 실제 최적화 로직은 없습니다.

생산량 변경의 인과 효과나 실제 절감액은 예측 성능만으로 보장되지 않습니다. KNN의 학습 오차 0은 자기 참조 특성이며, 학습·검증 차이 하나만으로 과적합을 확정하지 않습니다. 다른 데이터·설정으로 실행한다면 기존 HGB 선정 결론도 재평가해야 합니다.
