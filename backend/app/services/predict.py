"""批量推理 —— 丙负责，乙负责把它包成接口。

⚠️ 这个文件目前是**占位**，函数体是 NotImplementedError。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config import FAULT_NAMES


class PredictError(ValueError):
    """无法预测时抛出。API 层捕获转成 400。"""

    def __init__(self, message: str, code: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


@dataclass
class PredictionRow:
    row_index: int
    fault_class: int
    fault_name: str
    confidence: float
    probabilities: list[float]


@dataclass
class PredictResult:
    total_rows: int
    predictions: list[PredictionRow]
    elapsed_ms: float


def predict_batch(
    model,
    df: pd.DataFrame,
    feature_names: list[str],
    fill_values: dict[str, float] | None = None,
) -> PredictResult:
    """用已训练的模型批量预测。

    参数
    ----
    model
        已训练好的 sklearn 模型（有 predict 和 predict_proba）。
    df
        待预测数据，只含特征列，不含 fault。
    feature_names
        模型训练时的特征顺序。必须按这个顺序取列。
    fill_values
        训练时算出的缺失值填补值（每列中位数）。不传则用待预测数据自己的中位数。

    异常
    ----
    PredictError
        - code="DIMENSION_MISMATCH"：列数与模型不符
        - code="MISSING_COLUMN"：缺列
        - code="BAD_VALUE"：列里有非数字
    """
    import time

    import numpy as np

    started = time.perf_counter()

    check_dimension(df, feature_names)

    # 传进来的 feature_names 可能和模型实际训练时的不一致（调用方搞错），
    # 所以再拿模型自己的 n_features_in_ 兜一层，避免 sklearn 抛出难读的 ValueError。
    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and model_n != len(feature_names):
        raise PredictError(
            f"模型需要 {model_n} 个特征，但传入了 {len(feature_names)} 个特征名",
            code="DIMENSION_MISMATCH",
            context={"model_features": int(model_n), "given_features": len(feature_names)},
        )

    # 严格按训练时的顺序取列，不信任文件里的列序
    try:
        X = df[feature_names].astype(float)
    except (ValueError, TypeError) as exc:
        raise PredictError(
            f"待预测数据里有无法转成数字的值：{exc}",
            code="BAD_VALUE",
        ) from exc

    # 缺失值填补：优先用训练时记录的中位数，保证线上线下一致
    nan_cols = [c for c in feature_names if X[c].isna().any()]
    if nan_cols:
        for c in nan_cols:
            if fill_values and c in fill_values:
                val = fill_values[c]
            else:
                med = float(X[c].median())
                val = med if np.isfinite(med) else 0.0
            X[c] = X[c].fillna(val)

    X_arr = X.to_numpy()
    labels = model.predict(X_arr)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_arr)
        classes = list(getattr(model, "classes_", range(len(FAULT_NAMES))))
    else:  # 理论上不会走到，兜底
        probas = np.zeros((len(labels), len(FAULT_NAMES)))
        probas[np.arange(len(labels)), labels] = 1.0
        classes = list(range(len(FAULT_NAMES)))

    rows: list[PredictionRow] = []
    for i, label in enumerate(labels):
        cls = int(label)
        proba_full = _expand_proba(probas[i], classes)
        rows.append(
            PredictionRow(
                row_index=i,
                fault_class=cls,
                fault_name=FAULT_NAMES[cls] if 0 <= cls < len(FAULT_NAMES) else "未知",
                confidence=float(proba_full[cls]) if 0 <= cls < len(proba_full) else 0.0,
                probabilities=proba_full,
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return PredictResult(
        total_rows=len(rows),
        predictions=rows,
        elapsed_ms=round(elapsed_ms, 2),
    )


def _expand_proba(proba_row, classes: list[int]) -> list[float]:
    """模型可能只见过部分类别，按 classes_ 映射回 0–5 的位置。"""
    import numpy as np

    arr = np.asarray(proba_row, dtype=float)
    full = np.zeros(len(FAULT_NAMES), dtype=float)
    for pos, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls < len(FAULT_NAMES) and pos < len(arr):
            full[cls] = arr[pos]

    total = full.sum()
    if total > 0:
        full = full / total
    return [float(x) for x in full]


def check_dimension(df: pd.DataFrame, feature_names: list[str]) -> None:
    """预测前检查列是否对得上。"""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise PredictError(
            f"预测数据缺少 {len(missing)} 个特征列",
            code="MISSING_COLUMN",
            context={"missing": missing[:10]},
        )

    feature_cols = [c for c in df.columns if c != "fault"]
    if len(feature_cols) != len(feature_names):
        raise PredictError(
            f"预测数据的特征数与模型不符：模型需要 {len(feature_names)} 个，文件有 {len(feature_cols)} 个",
            code="DIMENSION_MISMATCH",
            context={
                "model_features": len(feature_names),
                "file_features": len(feature_cols),
            },
        )


def to_probability_list(proba_row) -> list[float]:
    """把一行概率转成 6 个 float，并归一化到和为 1。"""
    import numpy as np

    arr = np.asarray(proba_row, dtype=float)

    # 模型可能只见过部分类别，补齐到 6 类
    if len(arr) < len(FAULT_NAMES):
        padded = np.zeros(len(FAULT_NAMES), dtype=float)
        padded[: len(arr)] = arr
        arr = padded

    total = arr.sum()
    if total > 0:
        arr = arr / total

    return [float(x) for x in arr]
