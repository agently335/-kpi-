# -*- coding: utf-8 -*-
"""
分布式系统 KPI 故障诊断 —— 演示数据集生成脚本

重要声明:
    本脚本生成的全部数据均为【人工合成】，仅用于验证软件流程是否走通，
    不代表任何真实生产环境，不能据此宣称模型具备现实诊断能力。
    (对应验收场景 T05「信息透明」)

可复现性:
    - 随机种子固定为 SEED = 42
    - 重新运行本脚本，得到的数据与随附 CSV 完全一致
    - 生成方法、字段含义见同目录 数据字典.md

用法:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
SEED = 42
N_TRAIN = 500          # 训练数据行数
N_PREDICT = 50         # 待预测数据行数
MISSING_RATE = 0.01    # 训练集缺失值比例(模拟真实采集缺失)
MISSING_RATE_PRED = 0.005

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_DIR = os.path.join(OUT_DIR, "edge_cases")

# 6 类故障: 索引即标签值
FAULT_NAMES = {
    0: "正常",
    1: "CPU 饱和",
    2: "内存泄漏",
    3: "磁盘 IO 瓶颈",
    4: "网络异常",
    5: "服务异常/请求堆积",
}
FAULT_PROBS = [0.35, 0.15, 0.12, 0.14, 0.13, 0.11]

# 107 个特征按语义分为 7 组(仅用于生成与文档说明)
GROUPS = {
    "cpu":  (0, 12),    # feature0  - feature11   : CPU 相关
    "mem":  (12, 24),   # feature12 - feature23   : 内存相关
    "disk": (24, 39),   # feature24 - feature38   : 磁盘 IO 相关
    "net":  (39, 57),   # feature39 - feature56   : 网络相关
    "req":  (57, 72),   # feature57 - feature71   : 请求/吞吐相关
    "svc":  (72, 87),   # feature72 - feature86   : 服务健康/GC 相关
    "misc": (87, 107),  # feature87 - feature106  : 派生指标/其他
}
GROUP_ORDER = ["cpu", "mem", "disk", "net", "req", "svc", "misc"]
N_FEATURES = 107

# 每组特征的取值区间候选池(会循环分配给组内特征)
RANGE_POOL = {
    "cpu":  [(0.0, 100.0), (0.0, 100.0), (0.0, 16.0), (0.0, 16.0),
             (0.0, 60000.0), (0.0, 20000.0)],
    "mem":  [(0.0, 100.0), (0.0, 100.0), (0.0, 65536.0), (0.0, 4096.0),
             (0.0, 100.0), (0.0, 5000.0)],
    "disk": [(0.0, 100.0), (0.0, 500.0), (0.0, 8000.0), (0.0, 200000.0),
             (0.0, 1000.0)],
    "net":  [(0.0, 1000.0), (0.0, 100.0), (0.0, 10000.0), (0.0, 100.0),
             (0.0, 1000.0)],
    "req":  [(0.0, 5000.0), (0.0, 1000.0), (0.0, 100.0), (0.0, 5000.0),
             (0.0, 200.0)],
    "svc":  [(0.0, 100.0), (0.0, 50.0), (0.0, 1000.0), (0.0, 100.0),
             (0.0, 500.0)],
    "misc": [(0.0, 100.0), (0.0, 1000.0), (0.0, 10.0), (0.0, 500.0)],
}

# 每类故障在 7 个语义组上的“严重度”(0=无异常, 1=极严重)
# 刻意让部分故障共享相近的指标形态(如 CPU饱和 vs 服务异常、内存 vs 磁盘)，
# 使类别之间存在自然重叠 —— 否则数据完全可分，混淆矩阵就没有信息量了。
FAULT_SIGNATURE = {
    0: [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18],  # 正常
    1: [0.84, 0.30, 0.26, 0.28, 0.44, 0.38, 0.30],  # CPU 饱和
    2: [0.38, 0.87, 0.42, 0.26, 0.44, 0.58, 0.35],  # 内存泄漏
    3: [0.46, 0.42, 0.88, 0.28, 0.58, 0.44, 0.36],  # 磁盘 IO 瓶颈
    4: [0.28, 0.26, 0.26, 0.88, 0.72, 0.48, 0.34],  # 网络异常
    5: [0.52, 0.44, 0.32, 0.40, 0.86, 0.86, 0.40],  # 服务异常
}

# 小数据集(8 个具名特征)的定义: 列名 -> (所属组, 方向, 取值范围, 单位)
SMALL_SCHEMA = [
    ("cpu_usage_pct",   "cpu",  +1, (0.0, 100.0),   "%"),
    ("mem_usage_pct",   "mem",  +1, (0.0, 100.0),   "%"),
    ("disk_io_util_pct", "disk", +1, (0.0, 100.0),  "%"),
    ("disk_await_ms",   "disk", +1, (0.0, 500.0),   "ms"),
    ("net_latency_ms",  "net",  +1, (0.0, 1000.0),  "ms"),
    ("net_retrans_rate", "net", +1, (0.0, 1.0),     "比例"),
    ("req_qps",         "req",  -1, (0.0, 5000.0),  "次/秒"),
    ("error_rate",      "svc",  +1, (0.0, 1.0),     "比例"),
]
LABEL_COL = "fault"


# ----------------------------------------------------------------------------
# 核心生成逻辑
# ----------------------------------------------------------------------------
def build_feature_spec(rng):
    """为 107 个特征分配: 所属组、取值区间、响应方向(+1 正相关 / -1 负相关)"""
    specs = []
    for g in GROUP_ORDER:
        start, end = GROUPS[g]
        pool = RANGE_POOL[g]
        for j in range(end - start):
            lo, hi = pool[j % len(pool)]
            direction = -1 if rng.random() < 0.25 else +1
            specs.append((g, lo, hi, direction))
    assert len(specs) == N_FEATURES, f"特征数应为 {N_FEATURES}，实际 {len(specs)}"
    return specs


def sample_severity(rng, n):
    """为每个样本生成 7 个语义组的严重度向量"""
    labels = rng.choice(len(FAULT_PROBS), size=n, p=FAULT_PROBS)
    sev = np.array([FAULT_SIGNATURE[c] for c in labels], dtype=float)  # (n,7)
    # 加入个体差异 -> 造成类别间自然重叠，避免数据“过于干净”
    sev += rng.normal(0.0, 0.16, size=sev.shape)
    # 少量样本严重度整体偏弱(处于故障早期/边界态)，进一步制造混淆
    weak = rng.random(n) < 0.10
    sev[weak] = 0.18 + (sev[weak] - 0.18) * 0.45
    sev = np.clip(sev, 0.02, 0.98)
    return labels, sev


def severity_by_group(sev, group):
    return sev[:, GROUP_ORDER.index(group)]


def sev_to_ratio(s):
    """
    把严重度 s∈[0,1] 映射为“指标在 [lo,hi] 区间内的位置比例” r∈[0,1]。

    线性映射会让正常态(s≈0.18)落在区间 18% 处，例如 error_rate≈0.18(即 18%)，
    这对健康系统明显偏高。真实监控指标呈长尾分布：正常态紧贴基线，
    故障态才快速拉开。故用幂函数 r = s^2.4 做压缩：
        s=0.18 -> r≈0.017   (正常: error_rate 约 1.7%)
        s=0.88 -> r≈0.735   (严重故障)
    """
    return np.power(np.clip(s, 0.0, 1.0), 2.4)


def add_mult_noise(rng, base, lo, hi, sigma):
    """
    乘性(对数正态)噪声: value = base * exp(N(0, sigma^2)) + 小幅下限

    为什么不用加法噪声: 正常态基线贴近 lo(常为 0)，加法噪声会被下界截断，
    产生大量恰好等于 0 的单元格，看起来像"采集失败"而非"系统很闲"。
    乘性噪声天然为正且与量级成比例，更接近真实监控指标的抖动方式。
    """
    span = hi - lo
    floor = 0.008 * span                      # 极小正下限，避免严格等于 0
    val = base * np.exp(rng.normal(0.0, sigma, size=base.shape))
    val = val + floor * np.abs(rng.normal(0.0, 1.0, size=base.shape))
    return np.clip(val, lo + floor, hi)


def build_107(rng, specs, sev):
    """依据严重度向量生成 107 维特征矩阵"""
    n = sev.shape[0]
    X = np.zeros((n, N_FEATURES), dtype=float)
    for i, (g, lo, hi, direction) in enumerate(specs):
        s = severity_by_group(sev, g)
        r = sev_to_ratio(s)
        span = hi - lo
        # direction=+1: 指标随故障上升;  direction=-1: 指标随故障下降(如吞吐)
        base = lo + r * span if direction > 0 else lo + (1.0 - r) * span
        X[:, i] = add_mult_noise(rng, base, lo, hi, sigma=0.18)
    return X


def build_small(rng, sev):
    """依据同一套严重度向量生成 8 个具名特征(与 107 维文件描述同一批快照)"""
    n = sev.shape[0]
    out = {}
    for name, g, direction, (lo, hi), _unit in SMALL_SCHEMA:
        s = severity_by_group(sev, g)
        r = sev_to_ratio(s)
        span = hi - lo
        base = lo + r * span if direction > 0 else lo + (1.0 - r) * span
        out[name] = add_mult_noise(rng, base, lo, hi, sigma=0.16)
    return pd.DataFrame(out)


def inject_missing(df, rng, rate):
    """随机把部分单元格置为缺失(模拟真实采集中断)"""
    if rate <= 0:
        return df
    mask = rng.random(df.shape) < rate
    out = df.copy()
    out = out.mask(mask)
    return out


def to_107_frame(X):
    cols = [f"feature{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols)


# ----------------------------------------------------------------------------
# 自检: 手写 kNN 评估(不依赖 sklearn)
# ----------------------------------------------------------------------------
def knn_eval(Xtr, ytr, Xte, yte, k=5):
    # 缺失值用训练集均值填补
    mu = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    sd = np.nanstd(Xtr, axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    n_cls = len(FAULT_NAMES)
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
    idx = np.argsort(d2, axis=1)[:, :k]
    preds = np.array([np.bincount(ytr[row], minlength=n_cls).argmax() for row in idx])

    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(yte, preds):
        cm[t, p] += 1

    acc = np.trace(cm) / cm.sum()
    f1s = []
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return acc, float(np.mean(f1s)), cm


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    os.makedirs(EDGE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    specs = build_feature_spec(rng)

    # ---------- 训练数据(500 行) ----------
    labels, sev = sample_severity(rng, N_TRAIN)
    X107 = build_107(rng, specs, sev)
    df107 = to_107_frame(X107)
    df107 = inject_missing(df107, rng, MISSING_RATE)
    df107[LABEL_COL] = labels

    df_small = build_small(rng, sev)
    df_small = inject_missing(df_small, rng, MISSING_RATE)
    df_small[LABEL_COL] = labels

    # ---------- 待预测数据(50 行, 无标签) ----------
    _, sev_p = sample_severity(rng, N_PREDICT)
    X107p = build_107(rng, specs, sev_p)
    df107p = to_107_frame(X107p)
    df107p = inject_missing(df107p, rng, MISSING_RATE_PRED)

    df_smallp = build_small(rng, sev_p)
    df_smallp = inject_missing(df_smallp, rng, MISSING_RATE_PRED)

    # ---------- 写出 ----------
    p107 = os.path.join(OUT_DIR, "kpi_train_107.csv")
    psm = os.path.join(OUT_DIR, "kpi_train_small.csv")
    p107p = os.path.join(OUT_DIR, "kpi_predict_107.csv")
    psmp = os.path.join(OUT_DIR, "kpi_predict_small.csv")

    df107.to_csv(p107, index=False, float_format="%.4f")
    df_small.to_csv(psm, index=False, float_format="%.4f")
    df107p.to_csv(p107p, index=False, float_format="%.4f")
    df_smallp.to_csv(psmp, index=False, float_format="%.4f")

    # ---------- 边界/异常测试文件(T03/T04/T06) ----------
    head = df_small.head(20).copy()
    # 1) 缺一列(打乱了 schema)
    head.drop(columns=["disk_await_ms"]).to_csv(
        os.path.join(EDGE_DIR, "T03_missing_column.csv"),
        index=False, float_format="%.4f")
    # 2) 列顺序被打乱(但列名齐全)
    shuffled = head[["error_rate", "cpu_usage_pct", "req_qps", "net_latency_ms",
                     "mem_usage_pct", "disk_io_util_pct", "disk_await_ms",
                     "net_retrans_rate", LABEL_COL]]
    shuffled.to_csv(os.path.join(EDGE_DIR, "T03_shuffled_columns.csv"),
                    index=False, float_format="%.4f")
    # 3) 只有单一类别且行数很少
    single = df_small[df_small[LABEL_COL] == 0].head(12).copy()
    single.to_csv(os.path.join(EDGE_DIR, "T04_single_class.csv"),
                  index=False, float_format="%.4f")
    # 4) 只有表头没有数据
    df_small.head(0).to_csv(os.path.join(EDGE_DIR, "T06_empty.csv"),
                            index=False, float_format="%.4f")
    # 5) 含非法值
    bad = df_small.head(10).copy()
    # 先把数值四舍五入到 4 位，再转 object（转 object 后 float_format 不再生效）
    for col in bad.columns:
        bad[col] = bad[col].round(4).astype(object)
    bad.loc[bad.index[2], "cpu_usage_pct"] = "abc"        # 非数值
    bad.loc[bad.index[4], "mem_usage_pct"] = -50.0        # 负值(超物理范围)
    bad.loc[bad.index[6], "net_latency_ms"] = 99999999.0  # 离谱极值
    bad.loc[bad.index[8], "error_rate"] = ""              # 空字符串
    bad.to_csv(os.path.join(EDGE_DIR, "T06_bad_values.csv"), index=False,
               float_format="%.4f")

    # ---------- 自检 ----------
    print("=" * 64)
    print("演示数据集生成完成")
    print("=" * 64)
    print(f"输出目录: {OUT_DIR}\n")
    print(f"训练集 107 维 : {df107.shape[0]} 行 x {df107.shape[1] - 1} 特征 + 1 标签")
    print(f"训练集小样本  : {df_small.shape[0]} 行 x {df_small.shape[1] - 1} 特征 + 1 标签")
    print(f"待预测 107 维 : {df107p.shape[0]} 行 x {df107p.shape[1]} 特征(无标签)")
    print(f"待预测小样本  : {df_smallp.shape[0]} 行 x {df_smallp.shape[1]} 特征(无标签)")

    print("\n--- 标签分布(训练集) ---")
    vc = pd.Series(labels).value_counts().sort_index()
    for k_, v in vc.items():
        print(f"  故障{k_} {FAULT_NAMES[k_]}: {v:>3} 行  ({v / N_TRAIN * 100:5.1f}%)")

    miss = int(df107.isna().sum().sum())
    print(f"\n缺失单元格: {miss} / {df107.shape[0] * (df107.shape[1] - 1)} "
          f"({miss / (df107.shape[0] * (df107.shape[1] - 1)) * 100:.2f}%)")

    # 8:2 划分做质量自检
    Xo = df107.drop(columns=[LABEL_COL]).to_numpy()
    yo = labels
    n_tr = int(len(yo) * 0.8)
    acc, macro_f1, cm = knn_eval(Xo[:n_tr], yo[:n_tr], Xo[n_tr:], yo[n_tr:])

    print("\n--- 数据可分性自检 (前 400 行训练 / 后 100 行测试, kNN k=5) ---")
    print("  混淆矩阵(行=真实, 列=预测):")
    print("        " + "".join(f"  P{c}" for c in range(6)))
    for c in range(6):
        print(f"    T{c}  " + "".join(f"{v:4d}" for v in cm[c]))
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print("  (基线: 全猜多数类'正常'的 Accuracy = "
          f"{vc.max() / N_TRAIN:.4f}，但其 Macro-F1 仅 {2 * (vc.max() / N_TRAIN) / (1 + vc.max() / N_TRAIN):.4f})")

    print("\n--- 边界测试文件 ---")
    for fn in sorted(os.listdir(EDGE_DIR)):
        print(f"  edge_cases/{fn}")
    print()


if __name__ == "__main__":
    main()
