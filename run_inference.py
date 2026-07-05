import os
# HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 缓存路径
os.environ["HF_HOME"] = "D:/AI_Cache/huggingface"
import json
import time
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from prompts import get_prompt, PROMPT_TYPES
from answer_parser import extract_answer

# 模型名称
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# 生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 800,
    "do_sample": False, # 无随机，结果固定可复现
    # 开启采样后（"do_sample": True），这两个参数才会生效，用来控制回答多样性，
    # "temperature": 0.0,
    # "top_p": 1.0,
}


def load_questions():
    """加载300道题目"""
    questions = []
    with open("data/gsm8k_selected_questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def load_completed_keys(output_path):
    """读取已完成的任务键，用于断点续跑"""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    key = (record["question_id"], record["prompt_type"])
                    completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="启用断点续跑")
    parser.add_argument("--output", default="results/raw_outputs.jsonl", help="输出文件路径")
    parser.add_argument("--test_mode", action="store_true", help="测试模式：只跑前10题")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # 加载题目
    questions = load_questions()
    if args.test_mode:
        questions = questions[:10]
        print("测试模式：仅运行前10题")

    # 加载已完成任务
    completed_keys = load_completed_keys(args.output) if args.resume else set()
    print(f"已完成任务数: {len(completed_keys)}")

    # 4bit量化配置（省显存）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 加载模型和tokenizer
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成")

    # 遍历所有prompt类型和题目
    total_tasks = len(questions) * len(PROMPT_TYPES)
    pbar = tqdm(total=total_tasks, initial=len(completed_keys), desc="推理进度")

    with open(args.output, "a", encoding="utf-8") as out_f:
        for prompt_type in PROMPT_TYPES:
            for q in questions:
                qid = q["question_id"]
                task_key = (qid, prompt_type)

                # 断点续跑：跳过已完成的任务
                if task_key in completed_keys:
                    pbar.update(1)
                    continue

                # 构造prompt（使用官方chat template）
                prompt_text = get_prompt(prompt_type, q["question"])
                messages = [{"role": "user", "content": prompt_text}]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # 推理计时
                start_time = time.time()
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **GENERATION_CONFIG,
                        pad_token_id=tokenizer.eos_token_id
                    )

                end_time = time.time()
                inference_time = round(end_time - start_time, 3)

                # 提取生成的文本（去掉输入部分）
                input_len = inputs["input_ids"].shape[1]
                output_ids = outputs[0][input_len:]
                raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                output_token_count = len(output_ids)

                # 答案解析
                parse_result = extract_answer(raw_output, q["gold_answer_num"])

                # 构造记录
                record = {
                    "question_id": qid,
                    "original_qid": q["original_qid"],
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text,
                    "question": q["question"],
                    "gold_answer_raw": q["gold_answer_raw"],
                    "gold_answer_num": q["gold_answer_num"],
                    "raw_output": raw_output,
                    "normalized_answer": parse_result["normalized_answer"],
                    "extraction_status": parse_result["extraction_status"],
                    "is_normalized_correct": parse_result["is_normalized_correct"],
                    "is_strict_format_compliant": parse_result["is_strict_format_compliant"],
                    "is_strict_correct": parse_result["is_strict_correct"],
                    "output_token_count": output_token_count,
                    "inference_time_seconds": inference_time,
                    "model_name": MODEL_NAME,
                    "generation_config": GENERATION_CONFIG,
                    "parser_version": parse_result["parser_version"]
                }

                # 追加写入JSONL，立即刷新到磁盘
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_keys.add(task_key)
                pbar.update(1)

    pbar.close()
    print(f"推理完成，结果已保存到 {args.output}")


if __name__ == "__main__":
    main()