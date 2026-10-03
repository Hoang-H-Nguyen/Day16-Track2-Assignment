import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Cấu hình
# ============================================================

seed = 16


# ============================================================
# 2. Load dataset và đo thời gian load
# ============================================================

started = time.perf_counter()

df = pd.read_csv("creditcard.csv")

data_load_seconds = time.perf_counter() - started

# Tách features và target
X = df.drop(columns="Class")
y = df["Class"]


# ============================================================
# 3. Chia Train / Validation / Test
#
# 60% train
# 20% validation
# 20% test
# ============================================================

# Lần 1:
# 80% train+validation
# 20% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed,
    stratify=y,
)

# Lần 2:
# Validation = 25% của 80% = 20% toàn dataset
# Train = 75% của 80% = 60% toàn dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=seed,
    stratify=y_trainval,
)


# ============================================================
# 4. Tạo LightGBM model
# ============================================================

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=seed,
    n_jobs=2,
    verbosity=-1,
)


# ============================================================
# 5. Training + đo training time
# ============================================================

started = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(
            20,
            verbose=False,
        )
    ],
)

training_seconds = time.perf_counter() - started


# ============================================================
# 6. Predict trên TEST
# ============================================================

# Xác suất Class = 1 (fraud)
# Dùng cho AUC-ROC
probabilities = model.predict_proba(X_test)[:, 1]

# Chuyển xác suất thành nhãn 0/1
# threshold = 0.5
predictions = (probabilities >= 0.5).astype(int)


# ============================================================
# 7. Chuẩn bị đo inference
# ============================================================

# Một dòng để đo latency
one_row = X_test.iloc[:1]

# 1000 dòng để đo throughput
batch = X_test.iloc[:1000]


# Warm-up
# Không tính hai lần predict này vào kết quả benchmark
model.predict_proba(one_row)
model.predict_proba(batch)


# ============================================================
# 8. Hàm đo inference time
# ============================================================

def measured_seconds(data, repeats):
    elapsed = []

    for _ in range(repeats):
        started = time.perf_counter()

        model.predict_proba(data)

        elapsed.append(
            time.perf_counter() - started
        )

    # Dùng median để giảm ảnh hưởng của các lần chạy bất thường
    return float(np.median(elapsed))


# Đo 1 row 50 lần
single_seconds = measured_seconds(
    one_row,
    50,
)

# Đo batch 1000 rows 10 lần
batch_seconds = measured_seconds(
    batch,
    10,
)


# ============================================================
# 9. Tổng hợp kết quả
# ============================================================

result = {
    "recorded_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "architecture": platform.machine(),

    "versions": {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },

    # Dataset
    "dataset_rows": len(df),
    "fraud_rows": int(y.sum()),

    # Reproducibility
    "seed": seed,

    # Split
    "split": {
        "train": len(X_train),
        "validation": len(X_valid),
        "test": len(X_test),
    },

    # Model
    "n_jobs": 2,
    "decision_threshold": 0.5,

    # Timing
    "data_load_seconds": data_load_seconds,
    "training_seconds": training_seconds,

    # Early stopping
    "best_iteration": int(model.best_iteration_),

    # Evaluation metrics
    "auc_roc": float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    ),

    "accuracy": float(
        accuracy_score(
            y_test,
            predictions,
        )
    ),

    "f1": float(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    "precision": float(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    "recall": float(
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    # Latency
    "latency_1_row_ms": single_seconds * 1000,
    "latency_repeats": 50,

    # Throughput
    "batch_rows": len(batch),
    "batch_repeats": 10,

    "batch_1000_rows_seconds": batch_seconds,

    "throughput_1000_rows_per_second": (
        len(batch) / batch_seconds
    ),

    # Mô tả cách benchmark
    "timing_summary": (
        "median; warm-up excluded; "
        "predict_proba on pandas input"
    ),
}


# ============================================================
# 10. Lưu JSON
# ============================================================

Path("benchmark_result.json").write_text(
    json.dumps(
        result,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)


# ============================================================
# 11. In kết quả ra terminal
# ============================================================

print("\n===== LIGHTGBM BENCHMARK RESULT =====\n")

print(
    json.dumps(
        result,
        indent=2,
        allow_nan=False,
    )
)

print("\nSaved to benchmark_result.json")
