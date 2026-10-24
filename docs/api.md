# API 契约

> **这份文档是三人协作的唯一接口真相。**
> 后端按它实现、前端按它调用、测试按它验收。
> **W3 末冻结**。冻结后任何改动须三人同意并在此文件末尾留变更记录。

维护人：**乙**

---

## 全局约定

**基础地址**：`http://127.0.0.1:8000`

**格式**：请求与响应都是 JSON（上传文件除外，用 `multipart/form-data`）

**时间格式**：ISO 8601，如 `2026-10-24T15:03:16`

**错误响应统一结构**：

```json
{
  "detail": "缺少必要列：disk_await_ms",
  "code": "MISSING_COLUMN",
  "context": {
    "missing": ["disk_await_ms"],
    "expected": ["cpu_usage_pct", "mem_usage_pct", "..."]
  }
}
```

**错误码表**：

| HTTP | code | 什么时候 |
|---|---|---|
| 400 | `MISSING_COLUMN` | 缺少必要列 |
| 400 | `EXTRA_COLUMN` | 有多余列 |
| 400 | `COLUMN_ORDER` | 列顺序不符 |
| 400 | `EMPTY_FILE` | 文件没数据 |
| 400 | `BAD_VALUE` | 有非数字或非法值 |
| 400 | `SINGLE_CLASS` | 标签只有一类，无法训练 |
| 400 | `TOO_FEW_ROWS` | 行数太少 |
| 400 | `DIMENSION_MISMATCH` | 预测数据维度与模型不符 |
| 404 | `NOT_FOUND` | 数据集/模型不存在 |
| 500 | `INTERNAL` | 未预料的错误 |

---

## 故障类别（冻结）

| 编号 | 名称 |
|---|---|
| 0 | 正常 |
| 1 | CPU 饱和 |
| 2 | 内存泄漏 |
| 3 | 磁盘 IO 瓶颈 |
| 4 | 网络异常 |
| 5 | 服务异常/请求堆积 |

## 特征顺序（冻结）

**8 维版**：

```
cpu_usage_pct, mem_usage_pct, disk_io_util_pct, disk_await_ms,
net_latency_ms, net_retrans_rate, req_qps, error_rate
```

**107 维版**：`feature0` … `feature106`

**标签列名**：`fault`

---

# 1. GET /health

**用途**：确认后端活着。前端启动时先调这个。

**请求**：无参数

**响应 200**：

```json
{
  "status": "ok",
  "version": "0.1.0",
  "time": "2026-10-24T15:03:16.186934"
}
```

**前端怎么用**：

```javascript
try {
  const r = await fetch('http://127.0.0.1:8000/health')
  if (!r.ok) throw new Error('backend not ok')
  console.log('后端正常')
} catch (e) {
  console.error('后端没启动，请先运行 uvicorn')
}
```

**✅ 状态**：已实现，已实测通过。

---

# 2. POST /datasets

**用途**：上传一份带标签的 CSV，落盘 + 入库，返回数据集编号和行列数。

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 是 | CSV 文件，UTF-8 编码 |
| `has_label` | bool | 否 | 默认 `true`，训练数据必须带 `fault` 列 |

**响应 200**：

```json
{
  "dataset_id": 1,
  "filename": "kpi_train_small.csv",
  "rows": 500,
  "columns": 9,
  "message": "上传成功"
}
```

**错误示例**：

```json
{
  "detail": "缺少必要列：disk_await_ms",
  "code": "MISSING_COLUMN",
  "context": {"missing": ["disk_await_ms"]}
}
```

**验收要点（T03）**：
- 删一列的 CSV 必须报 `MISSING_COLUMN`，不能静默成功
- 列顺序打乱的 CSV 要么报 `COLUMN_ORDER`，要么自动重排并提示（二选一，写进契约就不许改）

---

# 3. GET /datasets/{id}/profile

**用途**：看这个数据集的详细画像——行列数、每列缺失率、标签分布、有没有列名问题。

**请求**：路径参数 `id`（整数）

**响应 200**：

```json
{
  "dataset_id": 1,
  "rows": 500,
  "columns": 9,
  "expected_features": 8,
  "missing_columns": [],
  "extra_columns": [],
  "column_order_ok": true,
  "label_present": true,
  "label_distribution": {"0": 178, "1": 68, "2": 63, "3": 64, "4": 71, "5": 56},
  "columns_detail": [
    {
      "name": "cpu_usage_pct",
      "dtype": "float64",
      "missing": 0,
      "missing_rate": 0.0,
      "minimum": 0.1,
      "maximum": 99.8,
      "mean": 42.3
    }
  ]
}
```

**响应 404**：

```json
{"detail": "数据集 99 不存在", "code": "NOT_FOUND", "context": {"dataset_id": 99}}
```

---

# 4. POST /trainings

**用途**：用指定的数据集训练一个模型，返回评估结果。模型落盘，版本入库。

**请求**：

```json
{
  "dataset_id": 1,
  "model_type": "random_forest",
  "test_size": 0.2,
  "random_seed": 42
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `dataset_id` | int | 是 | 必须已上传 |
| `model_type` | string | 否 | `random_forest`（默认）/ `logistic_regression` |
| `test_size` | float | 否 | 验证集比例，0.1–0.5，默认 0.2 |
| `random_seed` | int | 否 | 默认 42，**不许随机**，保证可复现 |

**响应 200**：

```json
{
  "model_id": 1,
  "dataset_id": 1,
  "model_type": "random_forest",
  "version": "v1",
  "trained_at": "2026-10-24T15:10:00",
  "train_rows": 400,
  "valid_rows": 100,
  "accuracy": 0.89,
  "macro_f1": 0.8848,
  "confusion_matrix": [
    [31, 1, 1, 0, 1, 1],
    [2, 14, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0],
    [1, 0, 0, 11, 0, 1],
    [0, 0, 0, 0, 16, 0],
    [1, 1, 1, 0, 0, 8]
  ],
  "per_class": [
    {"class_id": 0, "class_name": "正常", "precision": 0.8857, "recall": 0.8857, "f1": 0.8857, "support": 35}
  ],
  "feature_names": ["cpu_usage_pct", "mem_usage_pct", "disk_io_util_pct", "disk_await_ms", "net_latency_ms", "net_retrans_rate", "req_qps", "error_rate"]
}
```

**`confusion_matrix` 定义（易错，务必对齐）**：
- 6×6 二维数组
- **行 = 真实类别**，**列 = 预测类别**
- `confusion_matrix[2][3]` = 真实是"内存泄漏"、被预测成"磁盘 IO 瓶颈"的样本数
- 对角线 = 预测正确的样本数

**错误示例**：

```json
{"detail": "训练数据只包含 1 个类别，无法训练分类模型", "code": "SINGLE_CLASS", "context": {"classes_found": [0]}}
```

**验收要点（T01、T04）**：
- T01：用 `kpi_train_small.csv` 训练成功，返回完整混淆矩阵
- T04：用 `edge_cases/T04_single_class.csv` 必须报 `SINGLE_CLASS`，不能返回 100% 准确率骗人

---

# 5. GET /models

**用途**：列出所有训练过的模型，给前端做下拉框。

**请求**：可选查询参数 `dataset_id`（只列某个数据集的模型）

**响应 200**：

```json
{
  "total": 2,
  "items": [
    {
      "model_id": 2,
      "dataset_id": 1,
      "model_type": "logistic_regression",
      "version": "v2",
      "trained_at": "2026-10-24T15:20:00",
      "accuracy": 0.85,
      "macro_f1": 0.8412
    },
    {
      "model_id": 1,
      "dataset_id": 1,
      "model_type": "random_forest",
      "version": "v1",
      "trained_at": "2026-10-24T15:10:00",
      "accuracy": 0.89,
      "macro_f1": 0.8848
    }
  ]
}
```

排序：按 `trained_at` 倒序（新的在前）。

---

# 6. POST /predictions/batch

**用途**：上传一份不带标签的 CSV，用指定模型批量预测，返回每行的故障类别和置信度。

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 是 | 无标签 CSV，只含特征列 |
| `model_id` | int | 是 | 用哪个模型 |

**响应 200**：

```json
{
  "model_id": 1,
  "total_rows": 50,
  "predictions": [
    {
      "row_index": 0,
      "fault_class": 0,
      "fault_name": "正常",
      "confidence": 0.92,
      "probabilities": [0.92, 0.01, 0.02, 0.01, 0.02, 0.02]
    },
    {
      "row_index": 1,
      "fault_class": 3,
      "fault_name": "磁盘 IO 瓶颈",
      "confidence": 0.78,
      "probabilities": [0.05, 0.03, 0.06, 0.78, 0.04, 0.04]
    }
  ],
  "elapsed_ms": 23.5
}
```

**约束**：
- `probabilities` 长度必须是 6，且和为 1（浮点误差容许 ±0.001）
- `confidence` 等于 `probabilities[fault_class]`
- `row_index` 从 0 开始，对应输入文件的数据行（不含表头）

**错误示例**：

```json
{"detail": "预测数据的特征数与模型不符：模型需要 8 个，文件有 107 个", "code": "DIMENSION_MISMATCH", "context": {"model_features": 8, "file_features": 107}}
```

**验收要点（T02）**：
- 用 `kpi_predict_small.csv` + 8 维模型，返回 50 条预测
- 用 `kpi_predict_107.csv` + 8 维模型，必须报 `DIMENSION_MISMATCH`

---

# 7. 结果导出

导出不单独做接口，由前端把 `/predictions/batch` 的响应转成 CSV 下载。

**导出 CSV 的列（冻结）**：

```
row_index,fault_class,fault_name,confidence
```

---

## 变更记录

| 日期 | 改了什么 | 谁 | 三人同意 |
|---|---|---|---|
| 2026-10-24 | 初稿建立 | — | 待确认 |

> 每次改动追加一行。没有记录的改动视为违规。
