# -*- coding: utf-8 -*-
"""
基线模型快速验证脚本(不依赖 sklearn，纯 numpy 实现)

用途:
    验证演示数据集能否正常训练出有意义的模型，同时给团队一个
    「最小可跑通」的参考实现。等 W3 接入 sklearn 的 RandomForest 后，
    结果应当与此接近或更好。

用法:
    python baseline_knn.py
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "fault"
FAULT_NAMES = {0: "正常", 1: "CPU饱和", 2: "内存泄漏",
               3: "磁盘IO瓶颈", 4: "网络异常", 5: "服务异常"}


def load(path):
    return pd.read_csv(path)


def prepare(train_df, predict_df=None):
    """分离特征与标签，用训练集统计量做缺失填补 + 标准化"""
    y = train_df[LABEL].to_numpy().astype(int)
    X = train_df.drop(columns=[LABEL]).to_numpy(dtype=float)

    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    Xp = None
    if predict_df is not None:
        Xp = predict_df.to_numpy(dtype=float)
        Xp = np.where(np.isnan(Xp), mu, Xp)
        Xp = (Xp - mu) / sd
    return X, y, Xp


def knn_predict(Xtr, ytr, Xte, k=7):
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
    idx = np.argsort(d2, axis=1)[:, :k]
    n_cls = len(FAULT_NAMES)
    preds, confs = [], []
    for row in idx:
        votes = np.bincount(ytr[row], minlength=n_cls)
        preds.append(int(votes.argmax()))
        confs.append(float(votes.max() / k))
    return np.array(preds), np.array(confs)


def evaluate(y_true, y_pred, n_cls=6):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = np.trace(cm) / cm.sum()
    rows = []
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sup = cm[c, :].sum()
        rows.append((c, FAULT_NAMES[c], prec, rec, f1, sup))
    macro_f1 = float(np.mean([r[4] for r in rows]))
    return acc, macro_f1, cm, rows


def main():
    tr = load(os.path.join(HERE, "kpi_train_small.csv"))
    te = load(os.path.join(HERE, "kpi_predict_small.csv"))
    print(f"训练集: {tr.shape}   待预测集: {te.shape}\n")

    # 8:2 划分（固定切分，不用随机，方便对照）
    n = len(tr)
    cut = int(n * 0.8)
    Xtr_all, y_all, Xp = prepare(tr, te)

    acc, mf1, cm, rows = evaluate(y_all[cut:], knn_predict(Xtr_all[:cut], y_all[:cut], Xtr_all[cut:])[0])

    print("=" * 66)
    print("  8:2 留出验证结果（前 400 行训练 / 后 100 行验证）")
    print("=" * 66)
    print("  混淆矩阵（行=真实，列=预测）")
    print("        " + "".join(f"{'P' + str(c):>6}" for c in range(6)))
    for c in range(6):
        print(f"  T{c}    " + "".join(f"{v:>6d}" for v in cm[c]))
    print()
    print(f"  {'类别':<14}{'Precision':>11}{'Recall':>9}{'F1':>8}{'样本数':>8}")
    print("  " + "-" * 50)
    for c, name, p, r, f, s in rows:
        print(f"  {name:<14}{p:>11.4f}{r:>9.4f}{f:>8.4f}{s:>8d}")
    print("  " + "-" * 50)
    print(f"  Accuracy  = {acc:.4f}")
    print(f"  Macro-F1  = {mf1:.4f}\n")

    # 对无标签预测集输出结果（演示批量预测 + 导出）
    preds, confs = knn_predict(Xtr_all, y_all, Xp)
    out = te.copy()
    out["predicted_fault"] = preds
    out["fault_name"] = [FAULT_NAMES[p] for p in preds]
    out["confidence"] = np.round(confs, 4)
    out_path = os.path.join(HERE, "baseline_predictions.csv")
    out.to_csv(out_path, index=False, float_format="%.4f")

    print("=" * 66)
    print(f"  批量预测完成，已导出: {os.path.basename(out_path)}")
    print("=" * 66)
    dist = pd.Series(preds).value_counts().sort_index()
    for k_, v in dist.items():
        print(f"  {FAULT_NAMES[k_]:<14} {v:>3} 行 ({v / len(preds) * 100:5.1f}%)")
    print()
    print("  前 8 行预测结果：")
    print(out[["predicted_fault", "fault_name", "confidence"]].head(8).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
