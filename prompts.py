def get_prompt(prompt_type: str, question: str) -> str:
    """
    根据类型返回对应prompt
    prompt_type: P0 / P1 / P2 / P3
    """
    if prompt_type == "P0":
        # Direct Answer
        return f"""Solve the following math word problem.

Output only the final answer in the following format:
#### 

Problem:
{question}"""

    elif prompt_type == "P1":
        # Standard Zero-shot-CoT
        return f"""Solve the following math word problem step by step.

End with the final answer in the following format:
#### 

Problem:
{question}"""

    elif prompt_type == "P2":
        # Sequential CoT
        return f"""Solve the following math word problem step by step.
Organize your reasoning with a clear sequence:
first identify the needed information, then perform the calculation,
and finally give the answer.

End with the final answer in the following format:
#### 

Problem:
{question}"""

    elif prompt_type == "P3":
        # Logical-Connector CoT
        return f"""Solve the following math word problem step by step.
Use logical connectors where appropriate, such as "because" and
"therefore", to make the derivation clear.

End with the final answer in the following format:
#### 

Problem:
{question}"""

    else:
        raise ValueError(f"未知的prompt类型: {prompt_type}")

# 所有prompt类型列表
PROMPT_TYPES = ["P0", "P1", "P2", "P3"]