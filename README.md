# NLP-course-project
Level1 复现类 NLP 课程作业，基于 Qwen2.5-3B-Instruct、GSM8K 对比 4 类 Zero-shot-CoT 提示策略，包含推理、评测、统计检验、绘图完整流程

## 环境配置
安装GPU版本PyTorch
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
安装剩余所有依赖
```bash
pip install transformers==4.45.2 datasets==2.20.0 accelerate==0.32.1 bitsandbytes==0.43.3 statsmodels==0.14.2 pandas==2.2.2 numpy==1.26.4 seaborn==0.13.2 matplotlib==3.9.1 tqdm==4.66.4
```
在终端执行
```bash
set HF_ENDPOINT=https://hf-mirror.com
```
运行test_env.py验证环境
## 项目目录结构
```bash
nlp_cot_experiment/
├── data/                  # 存放300题固定抽样数据
│   ├── gsm8k_selected_ids.json
│   ├── gsm8k_selected_questions.jsonl
│   └── sampling_metadata.json
├── results/               # 全部实验输出结果
│   ├── raw_outputs.jsonl     # 原始逐条推理记录
│   ├── raw_outputs.csv
│   ├── metrics_summary.csv   # 各组指标汇总表
│   ├── significance_tests.csv# 统计检验p值结果
│   └── figures/           # 存放3个图表
├── tests/                 # 单元测试（可不运行）
├── prompts.py             # P0-P3四组提示词模板
├── sample_dataset.py      # 数据集抽样脚本
├── answer_parser.py       # 答案提取与归一化解析器
├── run_inference.py       # 核心模型推理脚本（支持断点续跑）
├── evaluate.py            # 指标计算
├── stats_analysis.py      # 统计检验
├── plot_results.py        # 可视化绘图
├── test_env.py            # 环境校验
└── README.md
```
## 运行顺序
- 如果要更改tokens限制长度上限，重新输出结果，可把results/raw_outputs.jsonl与results/raw_outputs.csv删除重新运行run_inference.py
- 1.运行sample_dataset.py，自动下载GSM8K数据集到D:\AI_Cache\huggingface\datasets，在项目data/生成300道抽样题目
- 2.运行run_inference.py，加载Qwen2.5-3B-Instruct模型，遍历300题×4种prompt=1200次推理，保存结果results/raw_outputs.jsonl，
- 3.运行evaluate.py，读取raw_outputs.jsonl，产出results/metrics_summary.csv（4组P0~P3汇总指标（准确率、格式率、平均 token、平均耗时）），results/raw_outputs.csv（推理记录表格）
- 4.运行stats_analysis.py，输出Cochran's Q总体检验、两两McNemar校正后p值，产出results/significance_tests.csv（存储所有 p 值与显著性标记）
- 5.运行plot_results.py，读取metrics_summary.csv绘制全部对比图表，保存到results/figures
