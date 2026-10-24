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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import (
    DEFAULT_MODEL,
    FAULT_NAMES,
    FEATURE_NAMES_107,
    FEATURE_NAMES_SMALL,
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
    fill_values: dict[str, float] = field(default_factory=dict)  # 缺失值填补用的中位数
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
    if model_type not in ("random_forest", "logistic_regression"):
        raise TrainingError(
            f"不支持的模型类型：{model_type}",
            code="BAD_VALUE",
            context={"model_type": model_type, "allowed": ["random_forest", "logistic_regression"]},
        )

    if not 0.1 <= test_size <= 0.5:
        raise TrainingError(
            f"验证集比例必须在 0.1–0.5 之间，收到 {test_size}",
            code="BAD_VALUE",
            context={"test_size": test_size},
        )

    # 顺序很重要：先查空表，再查值是否合法，最后才查能不能训练。
    # 反过来的话，一个"只有 10 行且含 abc"的文件会被误报成 TOO_FEW_ROWS，
    # 掩盖真正的错误原因。
    if df is None or len(df) == 0:
        raise TrainingError(
            "训练数据是空的（只有表头没有数据行）",
            code="EMPTY_FILE",
            context={"rows": 0},
        )

    X, y = split_xy(df)

    # 特征必须是纯数字。pandas 读进来若混了字符串，这里会失败。
    try:
        X = X.astype(float)
    except (ValueError, TypeError) as exc:
        raise TrainingError(
            f"特征列里有无法转成数字的值：{exc}",
            code="BAD_VALUE",
            context={"columns": list(X.columns)},
        ) from exc

    if len(X.columns) == 0:
        raise TrainingError(
            "训练数据里没有任何特征列（只有标签列）",
            code="MISSING_COLUMN",
            context={"missing": ["至少一个特征列"]},
        )

    check_trainable(y)

    feature_names = list(X.columns)

    # 缺失值用训练集的中位数填补（中位数比均值抗异常值）。
    # 这一步必须记录：P0 要求"保存预处理步骤"，否则线上预测会对不上。
    fill_values: dict[str, float] = {}
    nan_cols = [c for c in feature_names if X[c].isna().any()]
    if nan_cols:
        for c in nan_cols:
            med = float(X[c].median())
            if not np.isfinite(med):
                med = 0.0
            fill_values[c] = med
            X[c] = X[c].fillna(med)

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    # 固定的 8:2 切分。stratify 保证每类在训练/验证里都有，
    # random_state 保证同一个种子切出同样的划分。
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_arr,
        )
    except ValueError as exc:
        # stratify 在某类样本太少时会抛错，退化成不分层
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=random_seed,
        )

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        # 逻辑回归对量纲敏感：cpu_usage_pct 是 0-100，net_retrans_rate 是 0-1，
        # 不标准化会导致 lbfgs 不收敛（ConvergenceWarning）且精度明显变差。
        # 包成 Pipeline，这样模型对象自带标准化，预测时不用手动处理。
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_seed)),
            ]
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    cm = confusion_matrix(y_valid, y_pred)
    accuracy = float(np.trace(np.array(cm)) / np.array(cm).sum())
    per_class = _per_class_metrics(cm)
    macro_f1 = float(np.mean([m.f1 for m in per_class]))

    return TrainingResult(
        model_type=model_type,
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        train_rows=int(len(y_train)),
        valid_rows=int(len(y_valid)),
        confusion_matrix=cm,
        per_class=per_class,
        feature_names=feature_names,
        fill_values=fill_values,
        model_object=model,
    )


def _per_class_metrics(cm: list[list[int]]) -> list[ClassMetric]:
    """从混淆矩阵算出每一类的 precision / recall / F1。"""
    m = np.array(cm)
    out: list[ClassMetric] = []
    for c in range(len(FAULT_NAMES)):
        tp = int(m[c, c])
        fp = int(m[:, c].sum() - tp)
        fn = int(m[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(
            ClassMetric(
                class_id=c,
                class_name=FAULT_NAMES[c],
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=int(m[c, :].sum()),
            )
        )
    return out


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

    # 标签列可能读成字符串（比如含空格），统一转 int
    try:
        y = df[LABEL_COLUMN].astype(int)
    except (ValueError, TypeError) as exc:
        raise TrainingError(
            f"标签列 {LABEL_COLUMN} 里有无法转成整数的值：{exc}",
            code="BAD_VALUE",
            context={"column": LABEL_COLUMN},
        ) from exc

    X = df.drop(columns=[LABEL_COLUMN])

    # 列顺序按 config 里冻结的顺序重排，不信任文件里的列序。
    # 文件列序打乱（T03_shuffled_columns）时，只要列名齐全就能正确对上。
    known = _match_feature_set(X.columns)
    if known is not None:
        missing = [c for c in known if c not in X.columns]
        if missing:
            raise TrainingError(
                f"训练数据缺少 {len(missing)} 个特征列：{missing[:5]}",
                code="MISSING_COLUMN",
                context={"missing": missing, "expected": len(known)},
            )
        extra = [c for c in X.columns if c not in known]
        if extra:
            raise TrainingError(
                f"训练数据多了 {len(extra)} 个未知列：{extra[:5]}",
                code="EXTRA_COLUMN",
                context={"extra": extra, "expected": len(known)},
            )
        # 按冻结顺序重排
        X = X[known]

    # 检查标签合法性
    unknown = sorted(set(y.unique()) - set(range(len(FAULT_NAMES))))
    if unknown:
        raise TrainingError(
            f"标签里有未知类别：{unknown}，合法范围是 0-{len(FAULT_NAMES) - 1}",
            code="BAD_VALUE",
            context={"unknown_labels": unknown},
        )

    return X, y


def _match_feature_set(columns) -> list[str] | None:
    """判断这份数据属于哪套特征（8 维具名 / 107 维官方）。

    判据是"列名的交集谁更大"，而不是"列数刚好相等"。
    因为异常用例常常是删了一列或多了一列，列数对不上但仍应识别出是哪一套，
    这样才能报出 MISSING_COLUMN 而不是被当成一套全新的特征硬训练。
    """
    cols = set(columns)
    best: list[str] | None = None
    best_hit = 0
    for candidate in (FEATURE_NAMES_SMALL, FEATURE_NAMES_107):
        hit = len(cols & set(candidate))
        if hit > best_hit:
            best_hit = hit
            best = candidate
    # 至少要命中一半，否则认作"自定义特征"，不做列名校验
    if best is not None and best_hit * 2 >= len(best):
        return best
    return None


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
