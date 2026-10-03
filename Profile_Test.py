"""Profile 단독 비교. 다른 프로젝트 파일을 import하지 않습니다."""
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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
    x = make_features(data)
    profile = data[TARGETS].groupby([x.weekday, x.hour]).mean()
    return profile, data[TARGETS].mean()


def predict_with(fitted, data):
    profile, fallback = fitted
    x = make_features(data)
    keys = pd.MultiIndex.from_frame(x[["weekday", "hour"]])
    return profile.reindex(keys).fillna(fallback).to_numpy()

# 3. 이 모델만 시간순 3구간 검증
window = b - a
results = []
for label, start, end in [("CV1", a - 2 * window, a - window),
                          ("CV2", a - window, a), ("CV3", a, b)]:
    learning = df[df.index.normalize().isin(days[:start])]
    checking = df[df.index.normalize().isin(days[start:end])]
    if start <= 0 or learning.empty or checking.empty:
        raise ValueError("시간순 3구간 검증에 필요한 날짜가 부족합니다.")
    started = time.perf_counter()
    fitted = fit_model(learning)
    fit_seconds = time.perf_counter() - started
    predicted = predict_with(fitted, checking)
    scores = evaluate(checking[TARGETS], predicted)
    train_mae = evaluate(learning[TARGETS], predict_with(fitted, learning))["MAE"]
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        predict_with(fitted, checking)
        timings.append(time.perf_counter() - started)
    results.append({"split": label, **scores, "train_MAE": train_mae,
                    "gap_MAE": max(0, scores["MAE"] - train_mae),
                    "fit_seconds": fit_seconds,
                    "predict_ms_per_row": np.median(timings) * 1000 / len(checking)})
cv_results = pd.DataFrame(results).set_index("split")
summary = cv_results.mean()
summary["MAE_std"] = cv_results.MAE.std(ddof=1)
summary["worst_MAE"] = cv_results.MAE.max()
print("\n검증 구간별 결과\n", cv_results.round(4))
print("\n검증 요약\n", summary.round(4))

# 4. 기존 비교와 같은 학습 구간으로 최종 모델 학습 (테스트는 학습에 사용하지 않음)
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
