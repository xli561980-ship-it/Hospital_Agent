# 评测说明

## 当前评测脚本

评测入口为根目录 `evaluate_agent.py`。脚本支持两种主要模式：

- `--no-llm-run`：关闭图中的 LLM 路径，使用本地规则、TF-IDF 和相似度匹配评测结构化结果。
- 默认模式：走真实 LLM 混合链路，用于验证意图识别、槽位抽取和候选规则裁决。

常用参数：

| 参数 | 说明 |
|---|---|
| `--dataset` | 指定评测数据集路径 |
| `--include-profile` | 将年龄、性别、孕期信息拼入自然语言输入 |
| `--no-llm-run` | 跳过最终回复生成，评测结构化 State |
| `--judge-llm` | 使用 LLM-as-a-Judge 判断科室等价性 |
| `--limit` | 限制评测样本数量 |
| `--no-color` | 关闭终端颜色输出 |

## 数据集说明

当前仓库包含：

- `eval_dataset.json`：快速回归样本。
- `eval_dataset_stress.json`：口语化和压力样本。
- `eval_dataset_blind.json`：更偏泛化的盲测样本。
- `eval_dataset_blind_indomain.json`：知识库内盲测样本。
- `eval_dataset_200_each.json`：较完整的回归集合，覆盖位置、非急诊入口导诊、急诊入口提示和其他问题。

每条样本通常包含：

- `query`：用户原始问题。
- `expected_intent`：期望意图。
- `expected_department`：期望科室。
- `expected_location`：期望位置。
- `is_emergency_expected`：是否期望急诊入口提示。
- `age`、`gender`、`pregnancy_status`：可选用户画像字段。

## 指标说明

| 指标 | 含义 |
|---|---|
| 意图识别准确率 | 预测意图是否等于期望意图 |
| 科室推荐准确率 | 推荐科室是否匹配期望科室或可接受等价科室 |
| 位置检索准确率 | 位置结果或回复中是否包含期望位置 |
| 急症拦截成功率 | 期望急诊入口提示的样本是否被路由到急诊入口 |

这里的“急症拦截成功率”是评测口径名称，实际产品输出只做急诊入口提示，不判断病情严重程度。

## 如何运行

快速回归：

```bash
python evaluate_agent.py --dataset eval_dataset.json --no-llm-run --include-profile --no-color
```

完整离线回归：

```bash
python evaluate_agent.py --dataset eval_dataset_200_each.json --no-llm-run --include-profile --no-color
```

LLM 混合链路评测：

```bash
python evaluate_agent.py --dataset eval_dataset_200_each.json --include-profile --no-color
```

使用 LLM-as-a-Judge：

```bash
python evaluate_agent.py --dataset eval_dataset_200_each.json --include-profile --judge-llm --no-color
```

## 如何解读失败样本

失败样本应按以下顺序复盘：

1. 意图是否错分：查看 `expected_intent` 和 `predicted_intent`。
2. 槽位是否缺失：检查年龄、性别、孕期和症状是否从输入中正确抽取。
3. 候选规则是否召回：检查 `triage_candidate_rules` 是否包含正确规则。
4. 科室是否等价：确认预测科室是否为期望科室的合理上位或下位科室。
5. 位置是否命中：检查位置工具返回的服务、楼层、房间和路线。
6. 急诊入口提示是否触发：检查 `is_emergency` 和最终回复是否只提示急诊入口。
7. 安全边界是否生效：确认失败样本中没有诊断、治疗、用药或严重程度判断。

失败样本不要直接用单次结果修改大范围逻辑，应先判断是数据标注、知识库覆盖、召回排序、LLM 裁决还是安全后处理问题。

## 结果记录模板

不要手工编造结果。每次执行评测后，可将终端输出中的指标填写到下表：

| 日期 | 数据集 | 模式 | 意图识别准确率 | 科室推荐准确率 | 位置检索准确率 | 急症拦截成功率 | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| YYYY-MM-DD | `eval_dataset.json` | `--no-llm-run --include-profile` |  |  |  |  |  |
| YYYY-MM-DD | `eval_dataset_200_each.json` | `--no-llm-run --include-profile` |  |  |  |  |  |

