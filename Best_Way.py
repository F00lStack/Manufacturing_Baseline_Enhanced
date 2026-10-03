"""Best_Way: 기존 11개 방법의 다기준 비교에서 선정한 HGB. 뒤에 최적화 코드를 붙이세요."""
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# CSV를 이 코드와 같은 폴더에 두거나 DATA_PATH를 실제 파일 경로로 바꾸세요.
DATA_PATH = Path(os.environ.get("POWER_DATA_PATH", Path(__file__).with_name("okm_augumented_2021.csv")))
TARGETS = ["15분", "30분", "45분", "60분"]
WEATHER = ["기온", "풍속", "습도", "강수량"]
FEATURES = ["생산량", *WEATHER, "hour", "weekday", "month", "hour_sin", "hour_cos"]


def make_features(data):
    """DatetimeIndex와 생산량·날씨 열을 받아 학습과 같은 입력을 만듭니다."""
    x = data[["생산량", *WEATHER]].copy()
    if not isinstance(x.index, pd.DatetimeIndex) or x.index.hasnans:
        raise ValueError("시간은 유효한 pandas DatetimeIndex로 지정하세요.")
    x["hour"], x["weekday"], x["month"] = x.index.hour, x.index.dayofweek, x.index.month
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24)
    return x[FEATURES]


# 1. 이전 분석과 동일한 전처리
raw = pd.read_csv(DATA_PATH)
for col in ["날짜", "시간", *TARGETS, "생산량", *WEATHER, "전기요금(계절)"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
dates = pd.to_datetime(raw["날짜"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
valid_rows = (
    dates.notna() & raw["시간"].between(0, 23) & raw["시간"].mod(1).eq(0)
    & raw[TARGETS + ["생산량"]].notna().all(axis=1)
    & raw[TARGETS + ["생산량"]].ge(0).all(axis=1) & raw["전기요금(계절)"].gt(0)
)
raw["timestamp"] = dates + pd.to_timedelta(raw["시간"].where(valid_rows), unit="h")
duplicates = raw.loc[valid_rows, "timestamp"].duplicated(keep=False)
valid_rows.loc[duplicates.index[duplicates]] = False
df = raw.loc[valid_rows].set_index("timestamp").sort_index().copy()

# 2. 날짜 기준 70% / 15% / 15% 분할 (같은 날이 섞이지 않도록 함)
days = df.index.normalize().unique()
a, b = int(len(days) * 0.70), int(len(days) * 0.85)
train = df[df.index.normalize().isin(days[:a])].copy()
valid = df[df.index.normalize().isin(days[a:b])].copy()
test = df[df.index.normalize().isin(days[b:])].copy()
if min(len(train), len(valid), len(test)) == 0:
    raise ValueError("학습·검증·테스트를 나눌 날짜가 부족합니다.")


def evaluate(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean()),
            "R2": r2_score(actual, predicted)}

def fit_model(data):
    estimator = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=180, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1, early_stopping=False, random_state=42))
    model = make_pipeline(SimpleImputer(strategy="median"), estimator)
    model.fit(make_features(data), data[TARGETS])
    return model


def predict_with(fitted, data):
    return np.maximum(fitted.predict(make_features(data)), 0)

# 3. 기존 비교와 같은 학습 구간으로 최종 모델 학습 (테스트는 학습에 사용하지 않음)
model = fit_model(train)


def predict_power(data):
    """생산량·날씨와 DatetimeIndex를 받으며, 4개 전력 예측을 DataFrame으로 반환."""
    return pd.DataFrame(predict_with(model, data), index=data.index, columns=TARGETS)


test_prediction = predict_power(test)
test_metrics = evaluate(test[TARGETS], test_prediction)
print("\n테스트 결과:", {k: round(v, 4) for k, v in test_metrics.items()})

# ===== 이 아래에 후속 분석 또는 최적화 코드를 이어 붙이세요. =====
# 사용 가능: model, df, train, valid, test, FEATURES, TARGETS, predict_power
# candidate = test.iloc[:24].copy()
# candidate["생산량"] = candidate["생산량"] * 0.9
# predicted_power = predict_power(candidate)
