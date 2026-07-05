import re
from decimal import Decimal, InvalidOperation

# 解析器版本
PARSER_VERSION = "v1.0"

# 严格格式正则：最终行必须是 #### 数字
STRICT_PATTERN = r"^\s*####\s*[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*$"

# 宽容格式正则：允许Answer:前缀、空格、句点等
TOLERANT_PATTERN = (
    r"(?im)^\s*####\s*"
    r"(?:(?:final\s+answer|answer)\s*[:：]?\s*)?"
    r"([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?)"
    r"\s*[.。]?\s*$"
)


def normalize_number(num_str: str) -> str:
    """归一化数字：去千分位逗号、统一负号、转Decimal标准化"""
    if not num_str:
        return ""
    # 清理字符
    cleaned = num_str.strip().replace(",", "").replace("−", "-")
    try:
        # 用Decimal避免浮点误差，自动去掉末尾无意义的0
        num = Decimal(cleaned)
        # 整数转成整数形式，小数保留原有精度
        if num == num.to_integral_value():
            return str(int(num))
        else:
            return str(num)
    except InvalidOperation:
        return ""


def extract_answer(model_output: str, gold_answer_num: str) -> dict:
    """
    提取并判分答案，返回所有字段
    """
    lines = [line.rstrip() for line in model_output.strip().splitlines() if line.strip()]

    # 检查严格格式：最后一行是否完全匹配
    is_strict_format = False
    if lines:
        last_line = lines[-1]
        if re.fullmatch(STRICT_PATTERN, last_line):
            is_strict_format = True

    # 宽容提取：从后往前找最后一个####开头的行
    normalized_answer = ""
    extraction_status = "failed"

    # 倒序遍历找最后一个####行
    candidate_line = None
    for line in reversed(lines):
        if line.strip().startswith("####"):
            candidate_line = line
            break

    if candidate_line:
        match = re.search(TOLERANT_PATTERN, candidate_line)
        if match:
            raw_num = match.group(1)
            normalized_answer = normalize_number(raw_num)
            if normalized_answer:
                extraction_status = "strict" if is_strict_format else "tolerant"

    # 归一化标准答案
    gold_normalized = normalize_number(gold_answer_num)

    # 判分
    is_normalized_correct = (normalized_answer == gold_normalized) and (normalized_answer != "")
    is_strict_correct = is_normalized_correct and is_strict_format

    return {
        "normalized_answer": normalized_answer,
        "gold_normalized": gold_normalized,
        "is_normalized_correct": is_normalized_correct,
        "is_strict_format_compliant": is_strict_format,
        "is_strict_correct": is_strict_correct,
        "extraction_status": extraction_status,
        "parser_version": PARSER_VERSION
    }