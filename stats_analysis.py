import os
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
import itertools


def main():
    df = pd.read_csv("results/raw_outputs.csv")
    # 按题目+提示词分组，只保留最后一条记录
    df = df.drop_duplicates(subset=["question_id", "prompt_type"], keep="last")

    # 转换为宽格式：每行是一道题，每列是一个prompt的正确与否
    pivot = df.pivot(index="question_id", columns="prompt_type", values="is_normalized_correct")
    prompt_list = ["P0", "P1", "P2", "P3"]
    pivot = pivot[prompt_list]

    results = []

    # Cochran's Q 总体检验
    q_result = cochrans_q(pivot.values)
    results.append({
        "test": "Cochran's Q (Overall)",
        "comparison": "All 4 groups",
        "statistic": round(q_result.statistic, 4),
        "p_value": round(q_result.pvalue, 6),
        "significant": q_result.pvalue < 0.05
    })

    # 两两 McNemar 检验
    pairs = list(itertools.combinations(prompt_list, 2))
    p_values = []
    comparisons = []

    for a, b in pairs:
        # 构造2x2列联表
        table = pd.crosstab(pivot[a], pivot[b])
        # 确保表结构完整
        for val in [True, False]:
            if val not in table.index:
                table.loc[val] = [0, 0]
            if val not in table.columns:
                table[val] = 0
        table = table.sort_index().sort_index(axis=1)

        res = mcnemar(table.values, exact=False, correction=True)
        p_values.append(res.pvalue)
        comparisons.append(f"{a} vs {b}")
        results.append({
            "test": "McNemar (Pairwise)",
            "comparison": f"{a} vs {b}",
            "statistic": round(res.statistic, 4),
            "p_value": round(res.pvalue, 6),
            "significant_before_correction": res.pvalue < 0.05
        })

    # Holm 多重比较校正
    reject, p_adjusted, _, _ = multipletests(p_values, method="holm")

    # 回填校正结果
    for i, comp in enumerate(comparisons):
        for r in results:
            if r["comparison"] == comp and r["test"] == "McNemar (Pairwise)":
                r["p_value_holm"] = round(p_adjusted[i], 6)
                r["significant_after_holm"] = reject[i]
                break

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv("results/significance_tests.csv", index=False, encoding="utf-8-sig")

    print("统计检验完成：")
    print(f"Cochran's Q p值: {q_result.pvalue:.6f} {'(显著)' if q_result.pvalue < 0.05 else '(不显著)'}")
    print("\n两两比较(校正后):")
    for r in results[1:]:
        sig = "显著" if r["significant_after_holm"] else "不显著"
        print(f"  {r['comparison']}: 校正p={r['p_value_holm']:.6f} ({sig})")


if __name__ == "__main__":
    main()