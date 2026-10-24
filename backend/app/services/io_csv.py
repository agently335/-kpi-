"""CSV 读取与列校验 —— 乙（后端）负责实现，丙提供校验规则。

⚠️ 这个文件目前是**占位**，函数体是 NotImplementedError。

为什么先写空壳？
    T03 验收要求"删一列/列顺序打乱必须报错"。
    如果校验规则散落在各接口里，三个人写出来的错误提示会不一致。
    集中到这里，前端只要看 code 就知道怎么提示。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.config import FEATURE_NAMES_107, FEATURE_NAMES_SMALL, LABEL_COLUMN


class CsvValidationError(ValueError):
    """CSV 不合格时抛出。API 层捕获它，转成 400 + 错误码。"""

    def __init__(self, message: str, code: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing: int
    missing_rate: float
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


@dataclass
class CsvProfile:
    """CSV 画像。字段与 docs/api.md 的 GET /datasets/{id}/profile 对应。"""

    rows: int
    columns: int
    expected_features: int
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    column_order_ok: bool = True
    label_present: bool = False
    label_distribution: dict[str, int] = field(default_factory=dict)
    columns_detail: list[ColumnProfile] = field(default_factory=list)


# ------------------------------------------------------------------ 待实现
def read_csv(path) -> pd.DataFrame:
    """读 CSV，处理编码和空文件。

    异常
    ----
    CsvValidationError
        - code="EMPTY_FILE"：文件没有数据行
    """
    raise NotImplementedError("乙来实现：pd.read_csv + 空文件检查")


def validate_columns(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
    require_label: bool = True,
) -> tuple[list[str], bool]:
    """校验列名：缺列、多列、顺序。

    返回 (实际特征列名列表, 顺序是否正确)。

    异常
    ----
    CsvValidationError
        - code="MISSING_COLUMN"：缺列
        - code="EXTRA_COLUMN"：多列
    """
    raise NotImplementedError("乙来实现：比对列集合与顺序")


def profile_csv(df: pd.DataFrame, expected_features: list[str] | None = None) -> CsvProfile:
    """生成数据画像：行列数、每列缺失率、标签分布。"""
    raise NotImplementedError("乙来实现：遍历列统计 min/max/mean/missing")


def detect_schema(df: pd.DataFrame) -> list[str]:
    """自动判断是 8 维还是 107 维，返回对应的特征名列表。"""
    cols = set(df.columns) - {LABEL_COLUMN}
    if set(FEATURE_NAMES_SMALL).issubset(cols):
        return FEATURE_NAMES_SMALL
    if set(FEATURE_NAMES_107).issubset(cols):
        return FEATURE_NAMES_107
    missing_small = sorted(set(FEATURE_NAMES_SMALL) - cols)
    missing_107 = sorted(set(FEATURE_NAMES_107) - cols)
    raise CsvValidationError(
        "无法识别这份数据的特征列：既不是 8 维也不是 107 维",
        code="MISSING_COLUMN",
        context={
            "missing_8dim": missing_small,
            "missing_107dim_count": len(missing_107),
        },
    )
