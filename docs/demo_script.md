# Demo 脚本

本文档用于演示 Hospital Guide Agent 的核心场景。所有示例均基于当前 Mock Data，不代表真实医院信息。

## Demo 场景 1：普通科室推荐

用户输入：

```text
我25岁女，头疼挂什么科？
```

预期行为：

- 识别为导诊咨询。
- 抽取年龄、性别和主要症状。
- 从分诊规则知识库召回候选规则。
- 给出结构化科室推荐。
- 如信息不足，则只围绕导诊所需字段继续追问。

展示点：

- Intent Classification
- Slot Extraction
- Rule Retrieval
- LLM Semantic Selection 或本地规则回退
- Guardrailed Response

## Demo 场景 2：院内位置查询

用户输入：

```text
抽血在几楼？
```

预期行为：

- 识别为院内位置查询。
- 调用位置检索工具。
- 返回服务名称、楼栋、楼层、房间和路线。
- 不输出导诊或医学判断内容。

展示点：

- 位置查询意图识别
- `mock_data/locations.json` 检索
- Tool Calling
- 结构化位置回复

## Demo 场景 3：多轮号源查询

第一轮用户输入：

```text
我35岁男，胸口闷，想知道挂什么科？
```

第二轮用户输入：

```text
那什么时候有号？
```

预期行为：

- 第一轮完成导诊或急诊入口提示。
- 若推荐为普通门诊科室，第二轮沿用上一轮推荐科室查询排班。
- 若命中急诊入口提示，则不提供普通号源建议，只提示急诊分诊台。
- 排班结果来自 `mock_data/doctor_schedules.json`。

展示点：

- MemorySaver 多轮状态保持
- `current_phase` 从 `RECOMMENDED` 到 `SCHEDULE`
- 排班查询工具
- 急诊入口提示下的号源输出限制

## Demo 场景 4：医疗安全边界

用户输入：

```text
我是不是脑梗？吃什么药？
```

预期行为：

- 不进行疾病诊断。
- 不提供治疗建议。
- 不提供用药建议。
- 不判断病情严重程度。
- 将能力边界限定为科室推荐、院内位置、挂号和排班辅助。

展示点：

- Medical Safety Guardrails
- 禁止能力拦截
- 合规替代回复
- 最终输出不暴露内部推理链

