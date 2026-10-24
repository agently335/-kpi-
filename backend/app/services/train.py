"""训练与评估 —— 丙（模型/验证）负责实现。

⚠️ 这个文件目前是**占位**，函数体是 NotImplementedError。
   丙需要把 data/samples/baseline_knn.py 的逻辑搬进来，换成随机森林或逻辑回归。

为什么先写这个空壳？
    因为接口契约（docs/api.md 的 POST /trainings）已经定死了返回格式。
    先固定函数签名，丙实现时就不用猜"该返回什么"，
    乙写 API 时也不用猜"能拿到什么"。

实现要求：
    1. 必须用 RANDOM_SEED（42）固定随机种子，保证两次训练结果一致
    2. 必须按 FEATURE_NAMES 的顺序取特征，不能自动排序
    3. 混淆矩阵必须"行=真实，列=预测"
    4. 遇到单类别、行数太少，抛 TrainingError 并给出明确原因
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import (
    DEFAULT_MODEL,
    FAULT_NAMES,
    LABEL_COLUMN,
    RANDOM_SEED,
    TEST_SIZE,
)


class TrainingError(ValueError):
    """训练无法进行时抛出。API 层捕获它，转成 400 + 错误码。"""

    def __init__(self, message: str, code: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


@dataclass
class ClassMetric:
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TrainingResult:
    """训练结果。字段与 docs/api.md 的 POST /trainings 响应一一对应。"""

    model_type: str
    accuracy: float
    macro_f1: float
    train_rows: int
    valid_rows: int
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetric] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    model_object: object | None = None  # 真正的 sklearn 模型，存盘用，不返回给前端


# ------------------------------------------------------------------ 待实现
def train_model(
    df: pd.DataFrame,
    model_type: str = DEFAULT_MODEL,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> TrainingResult:
    """在带标签的数据上训练一个分类模型，返回评估结果。

    参数
    ----
    df
        训练数据。必须含 LABEL_COLUMN（"fault"）列，其余列是特征。
    model_type
        "random_forest" 或 "logistic_regression"。
    test_size
        验证集比例，0.1–0.5。
    random_seed
        随机种子。同一个种子 + 同一份数据 = 同一个结果。

    返回
    ----
    TrainingResult

    异常
    ----
    TrainingError
        - code="SINGLE_CLASS"：标签只有一类，无法训练
        - code="TOO_FEW_ROWS"：有效行数少于 20
        - code="BAD_VALUE"：特征里有非数字
    """
    raise NotImplementedError(
        "丙来实现：把 baseline_knn.py 的训练/评估逻辑搬到这里，"
        "换成 RandomForestClassifier 或 LogisticRegression。"
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """把 DataFrame 拆成 (特征 X, 标签 y)。

    特征列的顺序必须与 config.FEATURE_NAMES_* 一致。
    """
    if LABEL_COLUMN not in df.columns:
        raise TrainingError(
            f"训练数据缺少标签列：{LABEL_COLUMN}",
            code="MISSING_COLUMN",
            context={"missing": [LABEL_COLUMN]},
        )

    y = df[LABEL_COLUMN].astype(int)
    X = df.drop(columns=[LABEL_COLUMN])

    # 检查标签合法性
    known = set(range(len(FAULT_NAMES)))
    unknown = sorted(set(y.unique()) - known)
    if unknown:
        raise TrainingError(
            f"标签里有未知类别：{unknown}，合法范围是 0-{len(FAULT_NAMES) - 1}",
            code="BAD_VALUE",
            context={"unknown_labels": unknown},
        )

    return X, y


def check_trainable(y: pd.Series) -> None:
    """训练前的可行性检查。放在这里是为了让错误信息集中、好维护。"""
    n_classes = y.nunique()
    if n_classes < 2:
        raise TrainingError(
            f"训练数据只包含 {n_classes} 个类别，无法训练分类模型",
            code="SINGLE_CLASS",
            context={"classes_found": sorted(y.unique().tolist())},
        )
    if len(y) < 20:
        raise TrainingError(
            f"训练数据只有 {len(y)} 行，至少需要 20 行",
            code="TOO_FEW_ROWS",
            context={"rows": len(y)},
        )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """生成 6×6 混淆矩阵：行 = 真实，列 = 预测。

    不用 sklearn 的 confusion_matrix，因为要保证"行列方向"这一点
    在代码里是显式写出来的，不依赖库的默认行为。
    """
    k = len(FAULT_NAMES)
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m.tolist()
