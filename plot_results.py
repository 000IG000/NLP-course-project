import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns

# 用英文字体
plt.rcParams["axes.unicode_minus"] = False
plt_sns.set_style("whitegrid")

def main():
    metrics = pd.read_csv("results/metrics_summary.csv")
    os.makedirs("results/figures", exist_ok=True)

    # 归一化准确率对比图
    plt.figure(figsize=(8, 5))
    ax = plt_sns.barplot(data=metrics, x="prompt_type", y="normalized_accuracy", hue="prompt_type", palette="Blues_d", legend=False)
    plt.title("Normalized Accuracy of Different Prompts", fontsize=14)
    plt.xlabel("Prompt Type", fontsize=12)
    plt.ylabel("Normalized Answer Accuracy", fontsize=12)
    plt.ylim(0, 1)
    # 在柱子上标注数值
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{height:.1%}', ha="center")
    plt.tight_layout()
    plt.savefig("results/figures/accuracy_comparison.png", dpi=300)
    plt.close()

    # 格式合规率与严格准确率图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], metrics["strict_format_rate"], width, label="Strict Format Rate", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], metrics["strict_accuracy"], width, label="Strict Accuracy", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["prompt_type"])
    plt.title("Strict Format Rate & Strict Accuracy", fontsize=14)
    plt.ylabel("Ratio", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/figures/format_strict_comparison.png", dpi=300)
    plt.close()

    # 输出长度与推理耗时图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt_sns.barplot(data=metrics, x="prompt_type", y="avg_output_tokens", hue="prompt_type", palette="Greens_d", legend=False, ax=ax1)
    ax1.set_title("Average Output Tokens", fontsize=13)
    ax1.set_xlabel("Prompt Type")
    ax1.set_ylabel("Token Count")

    plt_sns.barplot(data=metrics, x="prompt_type", y="avg_inference_time", hue="prompt_type", palette="Reds_d", legend=False, ax=ax2)
    ax2.set_title("Average Inference Time", fontsize=13)
    ax2.set_xlabel("Prompt Type")
    ax2.set_ylabel("Seconds")
    plt.tight_layout()
    plt.savefig("results/figures/cost_comparison.png", dpi=300)
    plt.close()

    print("图表生成完成，已保存到 results/figures/")

if __name__ == "__main__":
    main()