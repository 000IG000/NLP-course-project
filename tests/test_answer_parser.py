import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from answer_parser import extract_answer


def test_all_cases():
    test_cases = [
        # (输出文本, 标准答案, 严格格式是否合规, 宽容是否正确)
        ("#### 99", "99", True, True),
        ("#### -1,200", "-1200", True, True),
        ("#### 12.5", "12.5", True, True),
        ("#### Answer: 99", "99", False, True),
        ("#### Final Answer: 99", "99", False, True),
        ("#### 99.", "99", False, True),
        ("The answer is 99", "99", False, False),
        ("Let's calculate.\n#### 42\nSome text", "42", False, True),
    ]

    all_pass = True
    for i, (output, gold, expect_strict, expect_correct) in enumerate(test_cases):
        result = extract_answer(output, gold)
        pass_strict = result["is_strict_format_compliant"] == expect_strict
        pass_correct = result["is_normalized_correct"] == expect_correct
        if not (pass_strict and pass_correct):
            print(f"用例{i}失败: 输出={output!r}")
            print(f"  预期: 严格={expect_strict}, 正确={expect_correct}")
            print(f"  实际: 严格={result['is_strict_format_compliant']}, 正确={result['is_normalized_correct']}")
            all_pass = False

    if all_pass:
        print("所有测试用例通过")
    else:
        print("部分测试用例失败")


if __name__ == "__main__":
    test_all_cases()