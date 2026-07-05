import os
import json
import pandas as pd


def main():
    # 读取JSONL
    records = []
    with open("results/raw_outputs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)

    # 按prompt分组计算指标
    metrics = df.groupby("prompt_type").agg(
        total_count=("question_id", "count"),
        normalized_correct=("is_normalized_correct", "sum"),
        normalized_accuracy=("is_normalized_correct", "mean"),
        strict_format_rate=("is_strict_format_compliant", "mean"),
        strict_accuracy=("is_strict_correct", "mean"),
        avg_output_tokens=("output_token_count", "mean"),
        avg_inference_time=("inference_time_seconds", "mean")
    ).reset_index()

    # 格式化百分比
    for col in ["normalized_accuracy", "strict_format_rate", "strict_accuracy"]:
        metrics[col] = metrics[col].round(4)

    metrics["avg_output_tokens"] = metrics["avg_output_tokens"].round(1)
    metrics["avg_inference_time"] = metrics["avg_inference_time"].round(3)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    metrics.to_csv("results/metrics_summary.csv", index=False, encoding="utf-8-sig")

    # 保存完整CSV方便后续分析
    df.to_csv("results/raw_outputs.csv", index=False, encoding="utf-8-sig")

    print("指标计算完成，结果如下：")
    print(metrics[["prompt_type", "normalized_accuracy", "strict_format_rate", "strict_accuracy"]])
    print(f"\n完整结果已保存到 results/metrics_summary.csv")


if __name__ == "__main__":
    main()