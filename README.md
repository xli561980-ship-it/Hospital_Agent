# 智能医院导诊台 Agent（Mock Demo）

这是一个面向售前演示的医院导诊 Agent Demo，定位是“院内导诊与挂号辅助”，不做诊断、不做治疗或用药建议。

它不是纯规则系统，也不是让模型自由发挥。当前真实链路是：

1. `LLM` 先做意图识别。
2. `LLM` 和规则一起抽取年龄、性别、孕期和症状。
3. 系统先从知识库里召回候选分诊规则。
4. `LLM` 在候选规则里做语义裁决，选择最匹配的 `rule_id`。
5. 安全层只允许输出知识库里已有的科室、位置和号源信息。

## 能力范围

- 多意图识别：科室推荐、院内位置指引、号源/排班查询、无关问题拒答
- 安全分诊：知识库候选召回 + `LLM` 语义裁决 + 规则兜底
- 工具调用：位置检索、科室匹配、医生排班查询
- 可降级运行：未配置 `LLM` Key 时，仍可使用本地规则和相似度匹配完成演示
- 量化评测：支持规则回归和 `LLM` 混合烟测

## 项目结构

- `agent.py`：基于 LangGraph 的多意图工作流
- `tools.py`：位置检索、科室匹配、排班查询工具
- `app.py`：Streamlit Web 对话 Demo
- `evaluate_agent.py`：离线评测脚本
- `mock_data/triage_rules.json`：分诊知识库
- `mock_data/locations.json`：院内位置知识库
- `mock_data/doctor_schedules.json`：医生排班 mock 数据
- `mock_data/routing_knowledge.json`：意图关键词、症状同义表达与派生短语

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

建议复制 `.env.example` 为 `.env` 并填写需要的 Key。

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Optional Gemini provider
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-1.5-flash

# Intent classification on/off
USE_INTENT_LLM=true

# Triage slot extraction + semantic rule selection on/off
USE_TRIAGE_LLM=true

# Optional deterministic demo/evaluation time
HOSPITAL_AGENT_NOW=2026-05-11T09:00:00
```

说明：

- `USE_INTENT_LLM=false` 时，意图分类走本地规则回退。
- `USE_TRIAGE_LLM=false` 时，分诊走本地规则匹配回退。
- 如果 `LLM` 不可用，代码会自动回退，不会把整个演示卡死。

## 运行

```bash
streamlit run app.py
```

## 评测

当前项目里有两类评测口径：

- `--no-llm-run`：只测离线回归，关闭 `LLM`，用于验证规则、字符相似度和兜底链路是否稳定
- 默认模式：走真实 `LLM` 混合链路，用于验证语义抽取和候选规则裁决是否正常

常用命令：

```bash
python evaluate_agent.py --dataset eval_dataset_200_each.json --no-llm-run --include-profile --no-color
python evaluate_agent.py --dataset eval_dataset_200_each.json --include-profile --no-color
```

评测集说明：

- `eval_dataset.json`：快速回归样本
- `eval_dataset_stress.json`：口语化和压力样本
- `eval_dataset_blind.json`：更偏泛化的盲测样本
- `eval_dataset_blind_indomain.json`：知识库内盲测样本
- `eval_dataset_200_each.json`：当前 800 条回归基线，覆盖位置、非急症分诊、急症分诊和其他问题

## 资料依据

本 Demo 的红旗症状词簇和同义表达只用于“就诊入口/科室路由”，不用于诊断。当前补充参考了公开权威资料和院内导诊常识，并外置到知识库文件中：

- CDC：卒中常见表现
- American Heart Association：心梗/胸痛警示表现
- CDC：川崎病常见表现
- CDC：糖尿病常见表现
- NIAMS / NHLBI / NIDDK / CDC：类风湿关节炎、干燥综合征、外周动脉疾病、血尿、认知障碍等常见表现

## 合规红线

Agent 被约束为：只能做科室推荐、急诊入口提示、院内位置指引和号源查询。

它不会：

- 进行疾病诊断
- 提供治疗或用药建议
- 判断病情严重程度
- 输出检查、手术、处方等医疗处置建议

真实医院落地前，必须由医疗、法务、安全和信息科共同审核规则、话术、日志、隐私与人工接管流程。
