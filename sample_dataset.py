import os
# HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 缓存路径
os.environ["HF_HOME"] = "D:/AI_Cache/huggingface"
import json
import random
from datasets import load_dataset


def main():
    # 固定随机种子
    random.seed(42)

    # 加载GSM8K官方测试集
    dataset = load_dataset("gsm8k", "main", split="test")
    all_data = list(dataset)

    # 随机抽取300题
    sampled_data = random.sample(all_data, 300)

    # GSM8K无原生qid，改用循环自定义唯一编号
    selected_ids = []
    selected_questions = []
    for idx, item in enumerate(sampled_data):
        qid_str = f"q_{idx:03d}"
        selected_ids.append(qid_str)
        selected_questions.append({
            "question_id": qid_str,
            "original_qid": qid_str,  # 数据集无原生id，复用自定义编号
            "question": item["question"],
            "gold_answer_raw": item["answer"],
            # 提取标准答案数值（GSM8K答案最后一行是 #### 数字）
            "gold_answer_num": item["answer"].split("####")[-1].strip()
        })

    # 保存文件
    os.makedirs("data", exist_ok=True)

    # 题目ID列表
    with open("data/gsm8k_selected_ids.json", "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, ensure_ascii=False, indent=2)

    # 题目详情JSONL
    with open("data/gsm8k_selected_questions.jsonl", "w", encoding="utf-8") as f:
        for item in selected_questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 抽样元数据
    metadata = {
        "dataset_name": "gsm8k",
        "split": "test",
        "sample_size": 300,
        "sampling_seed": 42,
        "sampling_method": "random_without_replacement",
        "dataset_revision": "main",
        "download_date": "2026-07-03",
        "note": "GSM8K数据集无原生qid，question_id为程序自定义编号"
    }
    with open("data/sampling_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"抽样完成，共抽取{len(selected_questions)}道题，已保存到data/目录")


if __name__ == "__main__":
    main()