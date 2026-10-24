"""请求 / 响应的数据结构（Pydantic 模型）。

这里定义的就是 **接口契约**。字段名一旦冻结，前端和测试都按它写。
改动必须三人同意并更新 docs/api.md。
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ================================================================ 健康检查
class HealthResponse(BaseModel):
    status: str = Field(description="ok 表示服务正常")
    version: str = Field(description="后端版本号")
    time: datetime = Field(description="服务器当前时间")


# ================================================================ 数据集
class DatasetUploadResponse(BaseModel):
    dataset_id: int
    filename: str
    rows: int = Field(description="数据行数，不含表头")
    columns: int = Field(description="列数")
    message: str = "上传成功"


class ProfileColumn(BaseModel):
    name: str
    dtype: str = Field(description="推断出的类型，如 float64 / object")
    missing: int = Field(description="缺失值个数")
    missing_rate: float = Field(description="缺失率 0-1")
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


class DatasetProfileResponse(BaseModel):
    dataset_id: int
    rows: int
    columns: int
    expected_features: int = Field(description="期望的特征数，8 或 107")
    missing_columns: list[str] = Field(default_factory=list, description="缺少的列")
    extra_columns: list[str] = Field(default_factory=list, description="多余的列")
    column_order_ok: bool = Field(description="列顺序是否与冻结顺序一致")
    label_present: bool = Field(description="是否含 fault 标签列")
    label_distribution: dict[str, int] = Field(
        default_factory=dict, description="各故障类别的样本数"
    )
    columns_detail: list[ProfileColumn] = Field(default_factory=list)


# ================================================================ 训练
class TrainingRequest(BaseModel):
    dataset_id: int
    model_type: str = Field(default="random_forest", description="random_forest / logistic_regression")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_seed: int = 42


class ClassMetrics(BaseModel):
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int = Field(description="该类真实样本数")


class TrainingResponse(BaseModel):
    model_id: int
    dataset_id: int
    model_type: str
    version: str = Field(description="模型版本号，如 v1")
    trained_at: datetime
    train_rows: int
    valid_rows: int
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]] = Field(description="行=真实，列=预测，6x6")
    per_class: list[ClassMetrics]
    feature_names: list[str] = Field(description="训练时用的特征顺序，必须与预测时一致")


class ModelSummary(BaseModel):
    model_id: int
    dataset_id: int
    model_type: str
    version: str
    trained_at: datetime
    accuracy: float
    macro_f1: float


class ModelListResponse(BaseModel):
    total: int
    items: list[ModelSummary]


# ================================================================ 预测
class PredictionItem(BaseModel):
    row_index: int = Field(description="原始数据中的行号，从 0 开始")
    fault_class: int = Field(ge=0, le=5, description="预测的故障类别 0-5")
    fault_name: str
    confidence: float = Field(ge=0.0, le=1.0, description="该预测的置信度")
    probabilities: list[float] = Field(description="6 个类别的概率，和为 1")


class BatchPredictionResponse(BaseModel):
    model_id: int
    total_rows: int
    predictions: list[PredictionItem]
    elapsed_ms: float = Field(description="推理耗时，毫秒")


# ================================================================ 错误
class ErrorResponse(BaseModel):
    detail: str = Field(description="给人看的错误说明")
    code: str = Field(description="机器可读的错误码，如 MISSING_COLUMN")
    context: dict = Field(default_factory=dict, description="额外信息，如缺了哪些列")
