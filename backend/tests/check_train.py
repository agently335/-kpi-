"""训练与推理的自检（丙的验收标准）。

跑法：
    cd backend
    python -m tests.check_train

断言：
    1. accuracy 落在 0.85–0.95
    2. 混淆矩阵是 6×6，所有元素相加 = 验证集行数
    3. 同一个种子连跑两次，结果完全一致
    4. 批量预测能跑通，概率和为 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import FAULT_NAMES, NUM_CLASSES  # noqa: E402
from app.services.predict import PredictError, predict_batch  # noqa: E402
from app.services.train import TrainingError, train_model  # noqa: E402

SAMPLES = BACKEND_DIR.parent / "data" / "samples"


def _fmt_cm(cm: list[list[int]]) -> str:
    header = "        " + "".join(f"{'P' + str(c):>6}" for c in range(len(cm)))
    lines = [header]
    for c, row in enumerate(cm):
        lines.append(f"  T{c}    " + "".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"[PASS] {name}" + (f" -> {detail}" if detail else ""))
        else:
            print(f"[FAIL] {name}" + (f" -> {detail}" if detail else ""))
            failures.append(name)

    train_df = pd.read_csv(SAMPLES / "kpi_train_small.csv")
    predict_df = pd.read_csv(SAMPLES / "kpi_predict_small.csv")

    # ---------------------------------------------------------- 训练
    result = train_model(train_df, random_seed=42)

    print("=" * 66)
    print("  8:2 留出验证结果")
    print("=" * 66)
    print(_fmt_cm(result.confusion_matrix))
    print()
    print(f"  {'类别':<16}{'Precision':>10}{'Recall':>9}{'F1':>8}{'样本数':>8}")
    print("  " + "-" * 52)
    for m in result.per_class:
        print(f"  {m.class_name:<16}{m.precision:>10.4f}{m.recall:>9.4f}{m.f1:>8.4f}{m.support:>8d}")
    print("  " + "-" * 52)
    print(f"  Accuracy  = {result.accuracy:.4f}")
    print(f"  Macro-F1  = {result.macro_f1:.4f}")
    print()

    check(
        "accuracy 在 0.85-0.95",
        0.85 <= result.accuracy <= 0.95,
        f"{result.accuracy:.4f}",
    )
    check(
        "macro_f1 在 0.80-0.95",
        0.80 <= result.macro_f1 <= 0.95,
        f"{result.macro_f1:.4f}",
    )

    cm = result.confusion_matrix
    check("混淆矩阵是 6x6", len(cm) == NUM_CLASSES and all(len(r) == NUM_CLASSES for r in cm))
    check(
        "混淆矩阵元素和 = 验证集行数",
        sum(sum(r) for r in cm) == result.valid_rows,
        f"{sum(sum(r) for r in cm)} vs {result.valid_rows}",
    )

    # 行和必须等于每类真实样本数（证明"行=真实"）
    row_sums_ok = all(sum(cm[c]) == result.per_class[c].support for c in range(NUM_CLASSES))
    check("行和等于每类真实样本数（行=真实）", row_sums_ok)

    # ---------------------------------------------------------- 可复现
    result2 = train_model(train_df, random_seed=42)
    check("同种子两次训练结果一致", result.accuracy == result2.accuracy and cm == result2.confusion_matrix)

    # ---------------------------------------------------------- 预测
    pred = predict_batch(
        result.model_object,
        predict_df,
        result.feature_names,
        fill_values=result.fill_values,
    )
    check("预测行数正确", pred.total_rows == len(predict_df), f"{pred.total_rows} 行")

    all_sum_one = all(abs(sum(p.probabilities) - 1.0) < 1e-6 for p in pred.predictions)
    check("每条概率和为 1", all_sum_one)

    all_valid_class = all(0 <= p.fault_class < NUM_CLASSES for p in pred.predictions)
    check("预测类别都在 0-5", all_valid_class)

    all_valid_conf = all(0.0 <= p.confidence <= 1.0 for p in pred.predictions)
    check("置信度都在 0-1", all_valid_conf)

    print()
    print("前 8 行预测结果：")
    print(f"  {'row':>4}{'类别':>6}  {'名称':<18}{'置信度':>8}")
    print("  " + "-" * 40)
    for p in pred.predictions[:8]:
        print(f"  {p.row_index:>4}{p.fault_class:>6}  {p.fault_name:<18}{p.confidence:>8.4f}")

    dist = {}
    for p in pred.predictions:
        dist[p.fault_class] = dist.get(p.fault_class, 0) + 1
    print()
    print("预测分布：")
    for c in sorted(dist):
        print(f"  {FAULT_NAMES[c]:<18}{dist[c]:>4} 行 ({dist[c] / pred.total_rows * 100:5.1f}%)")

    # ---------------------------------------------------------- 异常用例
    print()
    print("=" * 66)
    print("  异常输入用例（T03/T04/T06）")
    print("=" * 66)

    edge = BACKEND_DIR.parent / "data" / "edge_cases"
    expected = [
        ("T03_missing_column.csv", "MISSING_COLUMN"),
        ("T04_single_class.csv", "SINGLE_CLASS"),
        ("T06_empty.csv", "EMPTY_FILE"),
        ("T06_bad_values.csv", "BAD_VALUE"),
    ]
    for fname, want_code in expected:
        path = edge / fname
        try:
            train_model(pd.read_csv(path))
            check(f"{fname} 应该报 {want_code}", False, "居然训练成功了")
        except TrainingError as exc:
            check(f"{fname} 报 {want_code}", exc.code == want_code, f"实际 {exc.code}")

    # 列序打乱：列名齐全就应该能训练，且特征顺序被重排回冻结顺序
    try:
        shuffled = train_model(pd.read_csv(edge / "T03_shuffled_columns.csv"))
        check(
            "T03_shuffled_columns 列序被打乱也能训练",
            True,
            f"特征顺序 = {shuffled.feature_names[:3]}...",
        )
    except TrainingError as exc:
        check("T03_shuffled_columns 列序被打乱也能训练", False, f"报错 {exc.code}")

    # 预测阶段：缺列 / 维度不符
    bad_pred = predict_df.drop(columns=["net_latency_ms"])
    try:
        predict_batch(result.model_object, bad_pred, result.feature_names)
        check("预测缺列报 MISSING_COLUMN", False, "没报错")
    except PredictError as exc:
        check("预测缺列报 MISSING_COLUMN", exc.code == "MISSING_COLUMN", exc.code)

    # 维度不符：拿 8 维模型去喂只有 1 列的数据
    one_col = predict_df[["cpu_usage_pct"]]
    try:
        predict_batch(result.model_object, one_col, ["cpu_usage_pct"])
        check("预测维度不符报 DIMENSION_MISMATCH", False, "没报错")
    except PredictError as exc:
        check("预测维度不符报 DIMENSION_MISMATCH", exc.code == "DIMENSION_MISMATCH", exc.code)

    print()
    if failures:
        print(f"[RESULT] {len(failures)} 项失败：{failures}")
        return 1
    print("[RESULT] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
