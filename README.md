# KPI 故障诊断系统

> 教学项目：浏览器上传 KPI CSV → 训练基础分类器 → 展示故障类别与置信度 → 导出结果。
>
> **本项目不做**：真实分布式集群、生产日志采集、实时日志流、大型神经网络、竞赛级精度。

---

## 这是什么

一个最小的机器学习 Web 应用，把"训练模型"和"用模型"两件事用网页跑起来：

1. 上传一份带标签的 KPI 数据（CSV）
2. 点"训练"，后端选择随机森林或逻辑回归，划分训练/验证集
3. 查看混淆矩阵、Accuracy、Macro-F1
4. 上传一份不带标签的数据，批量预测，得到故障类别 0–5 和置信度
5. 导出结果为 CSV

**故障类别**：0 正常、1 CPU 饱和、2 内存泄漏、3 磁盘 IO 瓶颈、4 网络异常、5 服务异常/请求堆积

---

## 环境要求

| 工具 | 版本 | 用途 |
|---|---|---|
| Python | 3.10+ | 后端 |
| Node.js | 18+（可选） | 前端，只用 Python 也能跑通 API |

---

## 第一次运行（后端）

```powershell
# 1. 进入 backend 目录
cd backend

# 2. 装依赖（走清华镜像，国内快）
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 启动服务
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

看到这一行就成功了：

```
Uvicorn running on http://127.0.0.1:8000
```

然后浏览器打开 **http://127.0.0.1:8000/docs**，能看到自动生成的接口页面。

**验证健康检查**：

```powershell
curl http://127.0.0.1:8000/health
```

应返回：

```json
{"status":"ok","version":"0.1.0","time":"2026-10-24T15:00:00"}
```

> **按 Ctrl+C 停止服务。** 关掉窗口也会停止。

---

## 目录结构

```
kpi-diagnosis/
├── backend/
│   ├── main.py              # FastAPI 应用入口，注册路由
│   ├── requirements.txt     # Python 依赖
│   ├── app/
│   │   ├── config.py        # 常量：故障标签名、特征顺序
│   │   ├── schemas.py       # 请求/响应的数据结构
│   │   ├── routers/         # 各个接口的文件
│   │   │   ├── health.py
│   │   │   ├── datasets.py
│   │   │   ├── trainings.py
│   │   │   └── predictions.py
│   │   ├── services/        # 真正的业务逻辑（丙负责）
│   │   │   ├── io_csv.py    # 读 CSV、校验列
│   │   │   ├── train.py     # 训练 + 评估
│   │   │   └── predict.py   # 推理
│   │   └── db/              # SQLite 相关（乙负责）
│   └── tests/               # 测试用例
├── frontend/                # 前端（甲负责，W3 起搭建）
├── data/
│   ├── samples/             # 演示数据
│   └── edge_cases/          # 异常输入用例（T03/T04/T06）
├── docs/
│   └── api.md               # 5 个接口的契约（冻结后不得随意改）
└── README.md
```

---

## 演示数据

`data/samples/` 里有两套：

| 文件 | 说明 |
|---|---|
| `kpi_train_small.csv` | 500 行 × 8 个具名特征 + `fault` 标签，**推荐先用这个** |
| `kpi_train_107.csv` | 500 行 × feature0–feature106 + `fault`（对齐官方 107 维） |
| `kpi_predict_small.csv` | 50 行无标签，用来测试预测 |
| `kpi_predict_107.csv` | 同上，107 维版本 |

> ⚠️ **数据是合成的**，只用于验证流程，不代表任何真实系统。
> 汇报时必须标注"仅用于流程验证"（T05 要求）。

`data/edge_cases/` 里是故意做坏的输入，用来测格式校验：

| 文件 | 测什么 |
|---|---|
| `T03_missing_column.csv` | 少了一列（`disk_await_ms`） |
| `T03_shuffled_columns.csv` | 列顺序打乱 |
| `T04_single_class.csv` | 只有一个类别，无法训练 |
| `T06_empty.csv` | 只有表头，没有数据 |
| `T06_bad_values.csv` | 含 `abc`、负数、超大值、空串 |

**8 个特征的顺序（冻结，不得更改）**：

```
cpu_usage_pct, mem_usage_pct, disk_io_util_pct, disk_await_ms,
net_latency_ms, net_retrans_rate, req_qps, error_rate
```

---

## 常见问题

**Q：`python` 命令弹出微软商店？**
那是占位符，不是真 Python。装真 Python 后把 `WindowsApps` 从 PATH 里去掉，或用 `py` 代替 `python`。

**Q：`pip install` 卡住不动？**
在从美国服务器下载。按 `Ctrl+C`，然后加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 走清华源。

**Q：端口 8000 被占用？**
换个端口：`python -m uvicorn main:app --port 8001`。

**Q：克隆下来运行报 `ModuleNotFoundError`？**
没装依赖，执行 `python -m pip install -r requirements.txt`。

---

## 协作约定

- **W2 末冻结字段名**，W3 末冻结 5 个接口的请求/响应结构。冻结后改动须三人同意。
- 不提交 `*.db`、`*.pkl`、`__pycache__`、`node_modules`（已在 `.gitignore` 里）。
- 数据文件小于 5 MB 的可以提交，更大的走网盘。
- 每个接口至少 1 组请求/响应示例写在 `docs/api.md`，含错误情况。
