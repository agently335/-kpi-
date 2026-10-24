"""全局常量：故障标签、特征顺序、路径。

⚠️ 这两个列表是**冻结**的，改动会影响已训练的模型。
   任何改动必须三人同意并记录在 docs/api.md。
"""

from pathlib import Path

# ---------------------------------------------------------------- 项目路径
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EDGE_CASES_DIR = DATA_DIR / "edge_cases"
UPLOAD_DIR = PROJECT_DIR / "uploads"
MODEL_DIR = PROJECT_DIR / "models"

# ---------------------------------------------------------------- 版本号
APP_VERSION = "0.1.0"

# ---------------------------------------------------------------- 故障标签
# 顺序即类别编号，索引就是 fault 列的值
FAULT_NAMES: list[str] = [
    "正常",
    "CPU 饱和",
    "内存泄漏",
    "磁盘 IO 瓶颈",
    "网络异常",
    "服务异常/请求堆积",
]

NUM_CLASSES: int = len(FAULT_NAMES)

# ---------------------------------------------------------------- 特征顺序
# 8 维具名特征（演示数据用，推荐先用这套）
FEATURE_NAMES_SMALL: list[str] = [
    "cpu_usage_pct",
    "mem_usage_pct",
    "disk_io_util_pct",
    "disk_await_ms",
    "net_latency_ms",
    "net_retrans_rate",
    "req_qps",
    "error_rate",
]

# 107 维（对齐官方 feature0-feature106）
FEATURE_NAMES_107: list[str] = [f"feature{i}" for i in range(107)]

LABEL_COLUMN = "fault"

# ---------------------------------------------------------------- 训练参数
RANDOM_SEED = 42
TEST_SIZE = 0.2
DEFAULT_MODEL = "random_forest"


def ensure_dirs() -> None:
    """确保运行时需要的目录存在。"""
    for d in (UPLOAD_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
