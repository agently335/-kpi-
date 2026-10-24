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

    异常
    ----
    PredictError
        - code="DIMENSION_MISMATCH"：列数与模型不符
        - code="MISSING_COLUMN"：缺列
    """
    raise NotImplementedError("丙来实现：model.predict_proba + 组装 PredictionRow")


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
