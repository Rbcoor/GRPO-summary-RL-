# Summary-RL 本地模块说明

本文档记录 `src/summarizer` 下新增的本地辅助模块。后续如果新增模块、修改接口或调整参数，需要同步更新本文档。

## `keyword_reader.py`

### 作用

读取单个 Repliqa 文档 JSON 文件中的指定字段，并以纯文本形式返回。

### 类

`KeywordReader(json_file: str | Path)`

### 对外方法

`read(field_path: str) -> str`

根据字段名或点号路径读取内容。字符串会直接返回；如果读取到的是 `dict` 或 `list`，会转换成格式化后的 JSON 字符串返回。

支持的字段路径示例：

- `document_extracted`
- `document_topic`
- `questions`
- `questions.0.question`
- `questions.0.answer`

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/keyword_reader.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  document_extracted
```

### 参数

- `json_file`：单个文档 JSON 文件路径。
- `field`：要读取的字段名或点号字段路径。

## `paragraph_excutor.py`

### 作用

调用 `KeywordReader` 读取 `document_extracted`，按双换行符切分成段落，并按原文顺序编号。

### 类

`ParagraphExcutor(json_file: str | Path)`

### 对外方法

`execute() -> list[dict[str, Any]]`

返回格式：

```json
[
  {
    "paragraph_id": 1,
    "paragraph": "..."
  }
]
```

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/paragraph_excutor.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json
```

### 参数

- `json_file`：单个文档 JSON 文件路径。

## `paragraph_store.py`

### 作用

构建结构清晰的文档级 JSON，包含文档元信息、编号段落、文档内问题和答案。

### 类

`ParagraphStore(json_file: str | Path)`

### 对外方法

`questions() -> list[dict[str, str]]`

返回文档内所有问题及答案，包括：

- `question_id`
- `question`
- `answer`
- `long_answer`

`document_record() -> dict[str, Any]`

返回结构：

```json
{
  "document": {
    "document_id": "...",
    "document_topic": "...",
    "source_json": "...",
    "source_pdf": "..."
  },
  "paragraph_count": 0,
  "question_count": 0,
  "paragraphs": [],
  "questions": []
}
```

`save_json(output_path: str | Path) -> Path`

保存普通 JSON 文件。

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/paragraph_store.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  --output /tmp/kiqpsbuw.paragraph_store.json
```

### 参数

- `json_file`：单个文档 JSON 文件路径。
- `--output`：可选，输出 JSON 文件路径；不传时直接打印到终端。

## `dataset_sampler.py`

### 作用

面向整个导出的 Repliqa JSON 数据集，提供随机抽样和 batch 迭代能力。`batch_size` 的单位是“文档数量”，不是段落数量。

### 类

`DatasetSampler(dataset_root: str | Path, splits: list[str] | None = None, seed: int | None = None, shuffle: bool = True)`

### 对外方法

`sample(n: int) -> list[dict[str, Any]]`

随机抽取最多 `n` 个文档，并返回已切分段落后的文档记录。

`iter_batches(batch_size: int, limit: int | None = None) -> Iterator[list[dict[str, Any]]]`

按 batch 迭代文档。每个文档记录包含：

- `document_path`
- `document_id`
- `document_topic`
- `source_pdf`
- `paragraphs`

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/dataset_sampler.py \
  --split repliqa_0 \
  --batch-size 2 \
  --limit 3 \
  --seed 80
```

### 参数

- `--dataset-root`：包含 `repliqa_*` split 目录的数据集根目录。默认：`/tmp/repliqa_documents_by_file`。
- `--split`：指定要使用的 split，例如 `repliqa_0`。可以重复传入。
- `--batch-size`：每个 batch 的文档数量。
- `--limit`：可选，最多迭代多少个文档。
- `--seed`：可选随机种子。传入时可复现；不传时程序自动生成随机 seed，每次采样可能不同。
- `--no-shuffle`：不打乱，保持文件名排序。

## `paragraph_retriever.py`

### 作用

根据多个关键词，从单个文档中召回相关段落。召回采用混合检索策略：精确关键词命中、BM25 风格词项评分、轻量 fuzzy 匹配。结果按 `paragraph_id` 去重。

召回器会对 camelCase / PascalCase 关键词做拆分，例如将 `SolarEnergyInitiative` 作为 `solar energy initiative` 参与词项检索和短语匹配，以降低模型输出格式波动带来的漏召回。

### 类

`ParagraphRetriever(json_file: str | Path, fuzzy_threshold: float = 0.88)`

### 对外方法

`recall(keywords: list[str], min_score: float = 0.0) -> dict[str, Any]`

返回相关段落，每个段落包含：

- `paragraph_id`
- `paragraph`
- `hit_count`
- `exact_hit_count`
- `fuzzy_hit_count`
- `bm25_score`
- `score`
- `matched_keywords`
- `exact_hits`
- `fuzzy_hits`

### 动态 Top K 策略

`top_k` 始终由程序自动计算，不支持手动传入：

```text
top_k = ceil(paragraph_count * 0.2)
top_k 最小值 = 3
top_k 最大值 = 15
```

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/paragraph_retriever.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  WeTech sustainability community
```

### 参数

- `json_file`：单个文档 JSON 文件路径。
- `keywords`：一个或多个召回关键词。
- `--min-score`：可选，最低召回分数阈值。默认：`0.0`。
- `--fuzzy-threshold`：可选，fuzzy 匹配阈值。默认：`0.88`。

## `prompt_manager.py`

### 作用

统一管理 prompt 模板。当前首轮 prompt 用于从单篇文档文本中提取高质量关键词。

### 类

`PromptManager(keyword_min_count: int = 3, keyword_max_count: int = 7, summary_word_limit: int = 350)`

### 对外方法

`keyword_extraction_prompt(document_text: str) -> str`

构建关键词提取的 user prompt。

`keyword_extraction_messages(document_text: str) -> list[dict[str, str]]`

构建 OpenAI-style chat messages：

- `system`：定义模型为精确关键词提取助手。
- `user`：要求模型从文档中提取 3-7 个具体关键词。

`initial_summary_messages(round_json: str, document_id_json: str, recalled_paragraphs_json: str) -> list[dict[str, str]]`

构建首次摘要生成 prompt。首次摘要的特点：

- 只基于首次召回段落生成。
- 从零开始写第一版摘要。
- 重点保留事实、实体、日期、地点、原因、影响、行动和结果。
- 不提及“召回段落”。
- 输出不超过 350 词。

`submit_decision_messages(round_json: str, document_id_json: str, latest_summary_json: str, recalled_paragraphs_json: str) -> list[dict[str, str]]`

构建提交判断 prompt。判断环节要求模型输出严格 JSON：

```json
{
  "should_submit": true,
  "reason": "brief reason",
  "additional_keywords": []
}
```

如果摘要证据不足，则输出：

```json
{
  "should_submit": false,
  "reason": "brief reason",
  "additional_keywords": ["keyword one", "keyword two"]
}
```

`revision_summary_messages(round_json: str, document_id_json: str, latest_summary_json: str, recalled_paragraphs_json: str, latest_decision_json: str) -> list[dict[str, str]]`

构建后续摘要修订 prompt。它与首次摘要不同：

- 不是从零开始写。
- 必须保留上一版摘要中正确且有用的信息。
- 根据新召回段落补充缺失事实。
- 删除或修正不可靠、含糊、重复的内容。
- 不提及修订过程、召回段落或模型决策。
- 输出不超过 350 词。

### Prompt 要求

关键词提取 prompt 要求模型：

- 提取 3-7 个关键词。
- 避免泛词。
- 围绕文档核心特点。
- 优先提取命名实体、产品、地点、方法、事件、领域术语、核心概念。
- 覆盖文档的多个重要方面，例如主旨、原因或驱动因素、影响或后果、解决方案或建议、关键主体、地点、特色事件等。
- 不只提取最高频词；如果某个低频词代表文档中的重要信息，也应纳入候选。
- 避免重复或近义重复关键词。
- 使用自然可读短语；不要把多个词拼成 camelCase 或 PascalCase。
- 当文档包含多个重要方面时，优先接近 7 个关键词；只有文档本身非常单一时才输出较少关键词。
- 只返回 JSON 字符串数组，不输出解释。

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/prompt_manager.py \
  /tmp/test_document_text.txt
```

### 参数

- `document_file`：纯文本文件路径。
- `--min-count`：关键词最少数量。默认：`3`。
- `--max-count`：关键词最多数量。默认：`7`。

## `message_center.py`

### 作用

维护多轮摘要生成流程中的统一状态。消息中心记录系统中的关键信息，并提供模型可见状态，供后续 prompt 注入使用。

记录内容包括：

- 当前轮次 `current_round`
- 文档 ID
- 文档主题
- 文档原文
- 原始 JSON 路径
- 原始 PDF 路径
- 文档内 5 个问题及对应答案
- 多轮召回段落集
- 模型每轮生成的摘要
- 模型每轮是否决定提交
- 系统事件日志

注意：问题和答案只作为系统评估与状态记录信息，模型全程不可见，不应注入任何模型 prompt。

### 类

`MessageCenter(document_id: str, document_text: str, questions: list[dict[str, str]], document_topic: str | None = None, source_json: str | None = None, source_pdf: str | None = None)`

### 构造方法

`MessageCenter.from_document_json(json_file: str | Path) -> MessageCenter`

从一个导出的 Repliqa 文档 JSON 文件初始化消息中心。

### 对外方法

`set_round(round_id: int) -> None`

设置当前轮次。轮次不能小于 0。

`next_round() -> int`

当前轮次加 1，并返回新的轮次编号。

`add_recall_set(round_id: int, keywords: list[str], paragraphs: list[dict[str, Any]], source: str) -> None`

记录一轮召回结果。

`add_summary(round_id: int, summary: str, source: str) -> None`

记录模型生成的一版摘要。

`add_decision(round_id: int, should_submit: bool, reason: str, additional_keywords: list[str] | None = None, raw_output: str | None = None) -> None`

记录模型是否决定最终提交。如果不提交，可同时记录补充关键词。

`latest_summary() -> str | None`

返回最近一版摘要。

`latest_decision() -> dict[str, Any] | None`

返回最近一次提交决策。

`all_recalled_paragraphs() -> list[dict[str, Any]]`

汇总所有轮次召回段落，并按 `paragraph_id` 去重。

`model_visible_state() -> dict[str, Any]`

返回完整的模型可见信息。一般调试时使用；实际构造 prompt 时优先使用下面的单一消息接口，避免把无关上下文塞给模型。

`model_visible_json() -> str`

将完整模型可见信息序列化为 JSON 字符串。

#### 单一消息接口

下面每个接口只返回一种消息，方便上层 prompt 按需组合。

`round_message() -> dict[str, int]`

只返回当前轮次：`{"current_round": ...}`。

`round_json() -> str`

返回当前轮次的 JSON 字符串。

`document_id_message() -> dict[str, str]`

只返回文档 ID。

`document_id_json() -> str`

返回文档 ID 的 JSON 字符串。

`document_topic_message() -> dict[str, str | None]`

只返回文档主题。

`document_topic_json() -> str`

返回文档主题的 JSON 字符串。

`document_text_message() -> dict[str, str]`

只返回文档原文。

`document_text_json() -> str`

返回文档原文的 JSON 字符串。

`questions_message() -> dict[str, Any]`

只返回文档问题和答案。该接口仅供系统评估或调试使用，不应注入给模型。

`questions_json() -> str`

返回问题和答案的 JSON 字符串。该接口仅供系统评估或调试使用，不应注入给模型。

`recall_sets_message() -> dict[str, Any]`

只返回所有轮次的召回集合。

`recall_sets_json() -> str`

返回所有召回集合的 JSON 字符串。

`recalled_paragraphs_message(latest_only: bool = False) -> dict[str, Any]`

只返回召回段落。`latest_only=True` 时只返回最新一轮召回段落，否则返回所有轮次去重后的召回段落。

`recalled_paragraphs_json(latest_only: bool = False) -> str`

返回召回段落的 JSON 字符串。

`summaries_message() -> dict[str, Any]`

只返回所有摘要版本。

`summaries_json() -> str`

返回所有摘要版本的 JSON 字符串。

`latest_summary_message() -> dict[str, str | None]`

只返回最新摘要。

`latest_summary_json() -> str`

返回最新摘要的 JSON 字符串。

`decisions_message() -> dict[str, Any]`

只返回所有提交决策。

`decisions_json() -> str`

返回所有提交决策的 JSON 字符串。

`latest_decision_message() -> dict[str, Any]`

只返回最新提交决策。

`latest_decision_json() -> str`

返回最新提交决策的 JSON 字符串。

`system_state() -> dict[str, Any]`

返回完整系统状态，包括事件日志和源路径。

`save(output_path: str | Path) -> Path`

保存完整系统状态为 JSON 文件。

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/message_center.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  --output /tmp/message_center_state.json
```

只查看模型可见状态：

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/message_center.py \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  --model-visible
```

## `keyword_recall_pipeline.py`

### 作用

跑通关键词召回端到端流程：

1. 使用 `DatasetSampler` 从整个数据集中随机抽取一篇文档。
2. 使用 `ParagraphStore` 保存该文档的结构化段落 JSON。
3. 使用 `KeywordReader` 读取 `document_extracted`。
4. 使用 `PromptManager` 构造关键词提取 prompt。
5. 加载本地 `Qwen2.5-3B-Instruct` 模型完成关键词提取。
6. 使用 `ParagraphRetriever` 根据关键词召回相关段落。
7. 保存完整流程结果 JSON。

### 类

`KeywordRecallPipeline(dataset_root: str | Path, model_path: str | Path, output_dir: str | Path, splits: list[str] | None = None, seed: int | None = None, fuzzy_threshold: float = 0.88, max_new_tokens: int = 128)`

### 对外方法

`run() -> dict[str, Any]`

执行完整流程并返回结果。结果包含：

- `sampled_document`：随机抽取到的文档信息。
- `paragraph_store_path`：结构化段落 JSON 文件路径。
- `keyword_extraction`：模型抽取到的关键词和原始模型输出。
- `recall`：关键词召回结果。
- `result_path`：完整流程结果 JSON 文件路径。

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/keyword_recall_pipeline.py \
  --gpu 4 \
  --output-dir /tmp/keyword_recall_pipeline
```

### 参数

- `--dataset-root`：数据集根目录。默认：`/tmp/repliqa_documents_by_file`。
- `--model-path`：本地模型目录。默认：`/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct`。
- `--output-dir`：输出目录。默认：`/tmp/keyword_recall_pipeline`。
- `--split`：可选，指定数据集 split。可以重复传入。
- `--seed`：可选随机种子。传入时可复现；不传时程序自动生成随机 seed，每次采样可能不同。
- `--gpu`：使用的 GPU 编号。默认：`4`。
- `--fuzzy-threshold`：段落召回 fuzzy 匹配阈值。默认：`0.88`。
- `--max-new-tokens`：关键词生成最大 token 数。默认：`128`。

### 输出文件

每次运行会在 `--output-dir` 下生成两个文件：

- `<document_id>.paragraph_store.json`：该文档的结构化段落和问答信息。
- `<document_id>.keyword_recall_result.json`：完整流程结果，包括关键词和召回段落。

## `multi_round_summary_pipeline.py`

### 作用

执行多轮摘要生成流程。当前定义为：**每生成一次摘要才算一轮**。默认最大摘要轮次为 5 轮。

流程：

1. 随机抽取一篇文档。
2. 第一轮前，模型读取文档正文并生成初始关键词。
3. 系统根据关键词召回段落。
4. 生成摘要，当前摘要轮次 +1。
5. 模型读取最新摘要和召回段落，判断是否提交。
6. 如果提交，流程结束。
7. 如果不提交，模型必须补充新的关键词。
8. 系统用补充关键词召回新段落。
9. 下一轮模型基于上一版摘要和新召回段落生成修订摘要。
10. 重复判断和修订，直到提交或达到最大摘要轮次。

注意：模型全程看不到问题和答案。问题和答案只保留在消息中心中，供系统评估使用。

### 类

`MultiRoundSummaryPipeline(dataset_root: str | Path, model_path: str | Path, output_dir: str | Path, splits: list[str] | None = None, seed: int | None = None, max_summary_rounds: int = 5, fuzzy_threshold: float = 0.88)`

### 轮次规则

- `current_round` 表示当前摘要轮次。
- 只有生成摘要时才进入对应轮次。
- 初始关键词提取不计入摘要轮次。
- 最大轮次默认是 5。

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/multi_round_summary_pipeline.py \
  --gpu 4 \
  --max-summary-rounds 5 \
  --output-dir /tmp/multi_round_summary_pipeline
```

### 参数

- `--dataset-root`：数据集根目录。默认：`/tmp/repliqa_documents_by_file`。
- `--model-path`：本地模型目录。默认：`/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct`。
- `--output-dir`：输出目录。默认：`/tmp/multi_round_summary_pipeline`。
- `--split`：可选，指定数据集 split。可以重复传入。
- `--seed`：可选，随机种子；不传时使用默认随机流程。
- `--gpu`：使用的 GPU 编号。默认：`4`。
- `--max-summary-rounds`：最大摘要轮次。默认：`5`。
- `--fuzzy-threshold`：段落召回 fuzzy 匹配阈值。默认：`0.88`。

### 输出文件

每次运行会在 `--output-dir` 下生成：

- `<document_id>.multi_round_state.json`：消息中心完整状态。
- `<document_id>.multi_round_result.json`：多轮流程结果和 trace。
- `<document_id>.trajectory.json`：多轮流程轨迹，记录模型 messages、模型输出、系统召回步骤和指标。

## `trajectory_store.py`

### 作用

保存本地多轮摘要流程的轨迹。轨迹格式参考原项目 `rollout.py` 中 ART trajectory 的思想：记录模型可见 messages、模型输出、系统步骤、指标和奖励。

### 类

`TrajectoryStore(document_id: str, task_name: str = "multi_round_summary", metadata: dict[str, Any] = {})`

### 对外方法

`add_model_call(step_name: str, round_id: int, messages: list[dict[str, str]], output: str, parsed_output: Any | None = None, model_name: str | None = None) -> None`

记录一次模型调用，包括：

- 步骤名
- 轮次
- 模型看到的 messages
- 模型原始输出
- 解析后的输出
- 模型名称或路径

`add_system_step(step_name: str, round_id: int, payload: dict[str, Any]) -> None`

记录一次非模型系统步骤，例如段落召回。

`set_metric(name: str, value: Any) -> None`

设置单个指标。

`set_metrics(metrics: dict[str, Any]) -> None`

批量设置指标。

`to_dict() -> dict[str, Any]`

返回完整轨迹对象。

`save(output_path: str | Path) -> Path`

保存轨迹 JSON 文件。

### 轨迹结构

```json
{
  "task_name": "multi_round_summary",
  "document_id": "...",
  "metadata": {},
  "created_at": "...",
  "steps": [
    {
      "type": "model_call",
      "step_name": "initial_keyword_extraction",
      "round_id": 0,
      "messages": [],
      "output": "...",
      "parsed_output": []
    },
    {
      "type": "system_step",
      "step_name": "paragraph_recall",
      "round_id": 1,
      "payload": {}
    }
  ],
  "metrics": {},
  "reward": null
}
```

## `summary_judge.py`

### 作用

使用本地 judge 模型评测多轮流程生成的最终摘要质量。评测 prompt 参考原仓库 `rollout.py` 的两步设计，并且在实现上拆成两次模型调用，避免标准答案污染第一步回答：

1. 第一次调用只提供生成摘要和问题，让 judge 只依靠摘要回答每个参考问题；如果摘要无法回答，`generated_answer` 必须输出 `N/A`。
2. 第二次调用提供 `generated_answer` 和 reference answer，让 judge 判断 mostly match；匹配记为 `1`，否则记为 `0`。

评测 JSON 解析会优先读取纯 JSON，其次尝试 Markdown JSON 代码块和输出中的 JSON 对象；如果 `validation_benchmark.py` 中的 judge 输出仍然无法解析，会在当前文档目录写入 `<document_id>.judge_parse_error.json`，记录阶段名、错误信息和模型原始输出。

当前默认 judge 模型使用：

`/root/yaojiaxin/RL/models/Qwen3-14B`

评测阶段会读取 `message_center_state` 中保存的 5 个问题和答案，但这些问答只用于系统评测，不会进入摘要生成 prompt。

N/A 评判规则：

- 如果参考答案是 `N/A`、`NA`、`Not available`、`Not mentioned` 或等价表达，表示原文没有足够信息回答该问题。
- 如果 judge 在第一步根据摘要输出 `generated_answer: "N/A"`，则与 N/A reference answer 匹配，该题记为覆盖。
- 如果 judge 根据摘要生成了一个具体答案，但 reference answer 是 N/A，则该题不算覆盖，并应把该生成答案记入 `unsupported_claims`。
- N/A 问题不要求摘要包含一个不存在的具体事实。

最终训练奖励 `final_score` 不直接采用 judge 模型自己的综合分，而是由程序按固定权重计算：

```text
final_score =
  0.65 * answer_score
+ 0.15 * submit_score
+ 0.10 * round_score
+ 0.10 * length_score
```

其中：

- `answer_score = 正确覆盖的问题数 / 问题总数`
- `submit_score` 奖励合理自提交；低覆盖时提交不给分，高覆盖却不提交也不给分。
- `round_score` 奖励更少轮次完成，最大轮次为 1 时该项为 1。
- `length_score` 奖励 400-1200 字符的摘要；过短或过长都会降分。

### 类

`SummaryJudge(model_path: str | Path)`

### 对外方法

`evaluate(document_id: str, summary: str, questions: list[dict[str, Any]], max_new_tokens: int = 512) -> dict[str, Any]`

执行两次 judge 调用并返回结构化评测结果。

`answer_questions(document_id: str, summary: str, questions: list[dict[str, Any]], max_new_tokens: int = 512) -> dict[str, Any]`

第一次调用，只看摘要和问题，输出 `generated_answers`。

`score_answers(document_id: str, generated_answers: list[dict[str, str]], questions: list[dict[str, Any]], max_new_tokens: int = 512) -> dict[str, Any]`

第二次调用，将 `generated_answers` 与标准答案比较，输出逐题分数。

完整返回结构：

```json
{
  "answer_coverage": 0.0,
  "factual_consistency": 0.0,
  "completeness": 0.0,
  "conciseness": 0.0,
  "final_score": 0.0,
  "judge_model_final_score": 0.0,
  "reward_components": {},
  "reward_weights": {},
  "question_scores": [],
  "missing_evidence": [],
  "unsupported_claims": [],
  "answer_raw_output": "...",
  "score_raw_output": "...",
  "generated_answers": [],
  "judge_model": "..."
}
```

`load_summary_and_questions(result_json: str | Path) -> tuple[str, str, list[dict[str, Any]], Path | None]`

从 `<document_id>.multi_round_result.json` 中读取最终摘要、消息中心状态路径、问题答案和轨迹路径。

`update_trajectory_reward(trajectory_path: Path, judge_result: dict[str, Any]) -> None`

将程序计算后的 `final_score` 写回 `<document_id>.trajectory.json` 的顶层 `reward` 字段，并在 `metrics` 中补充：

- `judge_answer_coverage`
- `judge_factual_consistency`
- `judge_completeness`
- `judge_conciseness`
- `judge_final_score`
- `reward_answer_score`
- `reward_answered_questions`
- `reward_total_questions`
- `reward_submit_score`
- `reward_submitted`
- `reward_round_score`
- `reward_summary_rounds_used`
- `reward_max_summary_rounds`
- `reward_length_score`
- `reward_summary_chars`

### 命令行示例

```bash
CUDA_VISIBLE_DEVICES=4 /root/.conda/envs/summaryRL/bin/python src/summarizer/summary_judge.py \
  /tmp/multi_round_summary_pipeline/<document_id>.multi_round_result.json \
  --model-path /root/yaojiaxin/RL/models/Qwen3-14B \
  --update-trajectory-reward
```

输出文件默认保存为：

`<document_id>.multi_round_result.judge_result.json`

## `validation_benchmark.py`

### 作用

按照原仓库的验证集划分方式，固定抽取 `repliqa_0` 中的 91 篇文档作为验证集，批量运行当前多轮摘要流程并使用本地 judge 评测。

划分规则：

```text
files = sorted(repliqa_0/*.json)
random.Random(80).shuffle(files)
validation_files = files[:91]
```

该脚本会常驻加载两个模型，避免每篇文档重复加载权重：

- 摘要模型：默认 `/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct`
- judge 模型：默认 `/root/yaojiaxin/RL/models/Qwen3-14B`

摘要模型支持两种推理后端：

- `vllm`：默认后端，使用 vLLM 加速摘要侧关键词、摘要和提交判断生成。
- `transformers`：回退后端，使用 `AutoModelForCausalLM.generate`。

### 类

`ValidationBenchmark(...)`

### 主要参数

- `dataset_root`：数据集根目录，默认 `/tmp/repliqa_documents_by_file`
- `split`：默认 `repliqa_0`
- `split_seed`：验证集固定随机种子，默认 `80`
- `validation_size`：验证集数量，默认 `91`
- `max_summary_rounds`：最大摘要轮次，默认 `5`
- `summary_runner`：摘要推理后端，默认 `vllm`
- `summary_gpu_memory_utilization`：vLLM 摘要模型显存利用率，默认 `0.80`
- `summary_max_model_len`：可选，vLLM 摘要模型最大上下文长度
- `summary_batch_size`：摘要侧按文档批处理的 batch size，默认 `4`；设为 `1` 时回到逐文档顺序生成
- `summary_device`：摘要模型使用的可见 CUDA 设备，默认 `cuda:0`
- `judge_device`：judge 模型使用的可见 CUDA 设备，默认 `cuda:1`
- `limit`：可选，只跑前 N 篇，用于 smoke test
- `start_index`：可选，从验证集中的指定下标开始跑，用于断点续跑

### 输出

每篇文档会保存到：

`<output_dir>/<validation_index>_<document_id>/`

包含：

- `<document_id>.multi_round_result.json`
- `<document_id>.multi_round_state.json`
- `<document_id>.trajectory.json`
- `<document_id>.judge_result.json`

总报告：

- `<output_dir>/validation_items.jsonl`
- `<output_dir>/validation_report.json`

总报告会统计：

- `total_answered_questions`
- `mean_answered_questions`
- `mean_answer_score`
- `mean_final_score`
- `submit_count`
- `submit_rate`
- `mean_submit_round`
- `early_submit_count`
- `early_submit_rate`
- `mean_summary_rounds_used`
- `mean_summary_chars`

### 命令行示例

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/validation_benchmark.py \
  --gpus 1,4 \
  --summary-runner vllm \
  --summary-batch-size 4 \
  --max-summary-rounds 5 \
  --output-dir /tmp/summary_validation_benchmark
```

只跑前 3 篇做快速测试：

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/validation_benchmark.py \
  --gpus 1,4 \
  --summary-runner vllm \
  --summary-batch-size 4 \
  --max-summary-rounds 5 \
  --limit 3 \
  --output-dir /tmp/summary_validation_benchmark_smoke
```

### 摘要侧批处理

`validation_benchmark.py` 默认将多篇文档按阶段进行摘要侧 batch 推理：

1. 对 batch 内所有文档同时进行关键词提取。
2. 本地逐文档执行段落召回，并写入各自的 `MessageCenter` 与 `TrajectoryStore`。
3. 对仍在运行的文档同时生成本轮摘要。
4. 对这些文档同时执行提交判断。
5. 未提交且有补充关键词的文档进入下一轮，已提交或停止的文档退出后续轮次。

批处理只改变推理调度方式，不合并不同文档的消息中心或轨迹。每个文档仍独立保存 `document_id`、`round_id`、召回段落、摘要、提交判断和最终 reward。

关键词提取阶段的生成上限为 `256` tokens，用于降低模型返回 JSON 数组时被截断导致解析失败的概率。

如果关键词模型异常重复输出导致 JSON 数组被截断，验证脚本会从已生成文本中提取完整引号字符串，按大小写无关方式去重，并最多保留 7 个关键词继续流程。

### 启动脚本

`run_validation_vllm.sh`

从仓库根目录启动固定 91 篇验证集评测，摘要侧使用 vLLM，judge 侧使用本地 Qwen3-14B：

```bash
./run_validation_vllm.sh
```

常用环境变量：

- `OUTPUT_DIR`：输出目录，默认 `/tmp/summary_validation_benchmark_91_vllm_r5`
- `GPUS`：可见 GPU，默认 `1,4`
- `MAX_SUMMARY_ROUNDS`：最大摘要轮次，默认 `5`
- `SUMMARY_BATCH_SIZE`：摘要侧按文档批处理的 batch size，默认 `4`
- `LIMIT`：只跑前 N 篇，用于 smoke test
- `START_INDEX`：从验证集指定下标开始跑
- `SUMMARY_GPU_MEMORY_UTILIZATION`：vLLM 摘要模型显存利用率，默认 `0.80`

例如只跑 1 篇：

```bash
LIMIT=1 OUTPUT_DIR=/tmp/summary_validation_vllm_smoke ./run_validation_vllm.sh
```

## `art_trajectory_adapter.py`

### 作用

将 `multi_round_summary_pipeline.py` 保存的 `<document_id>.trajectory.json` 转换为 ART 可训练的 `art.Trajectory` 对象。

每次模型调用原本都是独立的 prompt/response，因此适配器将每个摘要生成步骤分别转换为一个 ART 轨迹，而不是将关键词提取、摘要生成和提交判断拼成不连续的单一对话。默认只适配 `initial_summary` 和 `revision_summary`，即真正需要优化的摘要动作。

问题和答案不会由本模块读取或写入模型消息。它们仅应由外部奖励评估器使用，以计算最终摘要的 `reward`，再传入本模块。

## `train_saved_trajectories_smoke.py`

### 作用

从已经保存的验证轨迹中抽取若干篇，构造 ART `TrajectoryGroup`，并用 `LocalBackend` 启动一次本地训练调用，用来验证保存轨迹是否能进入 ART 训练链路。

该脚本默认只取每篇文档的 `initial_summary` 作为训练样本，并把多篇文档放入同一个 smoke group 中，以保证 group 内存在 reward 方差。由于这些轨迹来自离线保存的模型输出，没有原始 logprobs，训练调用会设置 `allow_training_without_logprobs=True`。

脚本会通过 ART 的 `_internal_config` 给本地 Unsloth/vLLM 后端传入更保守的默认训练参数：`max_seq_length=8192`、`gpu_memory_utilization=0.45`、`per_device_train_batch_size=1`，并显式设置 `engine_args.enable_sleep_mode=False`。这是为了先验证轨迹能否启动训练，避免 32K 上下文、高 vLLM 显存占用以及 sleep/standby allocator 在 smoke test 阶段触发 CUDA graph OOM 或 MemPool 冲突。

### 命令行示例

```bash
CUDA_VISIBLE_DEVICES=1 /root/.conda/envs/summaryRL/bin/python train_saved_trajectories_smoke.py \
  --trajectory-root /tmp/summary_validation_benchmark_91_vllm_r5_batch4 \
  --limit 10 \
  --clean
```

### 参数

- `--trajectory-root`：包含验证结果子目录的根目录。
- `--limit`：使用前多少篇轨迹，默认 `10`。
- `--base-model`：ART 训练使用的基础模型，默认 `/root/yaojiaxin/RL/models/Qwen2.5-3B-Instruct`。
- `--art-path`：本地 ART 输出目录，默认 `/tmp/summary_art_smoke`。
- `--learning-rate`：ART 训练学习率，默认 `5e-6`。
- `--max-seq-length`：本地后端最大序列长度，默认 `8192`。
- `--gpu-memory-utilization`：Unsloth/vLLM 可使用的 GPU 显存比例，默认 `0.45`。
- `--per-device-train-batch-size`：单卡训练 batch size，默认 `1`。
- `--gradient-accumulation-steps`：梯度累积步数，默认 `1`。
- `--server-port`：ART 本地 OpenAI 兼容 vLLM 服务端口，默认 `18000`，用于避开常见的 `8000` 端口占用。
- `--clean`：运行前清理 ART smoke 输出目录。

### 类

`ARTTrajectoryAdapter(trajectory_file: str | Path)`

### 对外方法

`build(reward: float | None = None, trainable_steps: Iterable[str] = ("initial_summary", "revision_summary")) -> list[art.Trajectory]`

读取轨迹文件，返回 ART 训练所需的轨迹列表。每个返回对象的 `messages_and_choices` 由保存的模型输入 messages 与一个可训练的 `Choice` 输出组成。

- `reward`：训练奖励。若不传则读取保存轨迹的顶层 `reward`；两者都不是数值时会报错，避免用未评估的样本训练。
- `trainable_steps`：需要转为训练样本的步骤名列表。可加入 `initial_keyword_extraction` 或 `submit_decision` 以训练其他动作，但默认只训练摘要动作。

`build_group(reward: float | None = None, trainable_steps: Iterable[str] = ("initial_summary", "revision_summary")) -> art.TrajectoryGroup`

将本次工作流中选中的 ART 轨迹封装为一个 `TrajectoryGroup`，便于后续交给 ART 的训练接口。

### 命令行验证

```bash
/root/.conda/envs/summaryRL/bin/python src/summarizer/art_trajectory_adapter.py \
  /tmp/multi_round_trajectory_test/uaoxkrne.trajectory.json \
  --reward 0.8
```

命令会输出 ART 日志视图，其中 assistant 输出应标记为 `"trainable": true`。该命令仅验证和预览；实际训练时应在 Python 代码中调用 `build()` 或 `build_group()`，将返回的 ART 对象传给 `art.gather_trajectory_groups` / `model.train`。

## Shell 辅助脚本

### `run_keyword_reader.sh`

从仓库根目录运行 `keyword_reader.py`。

```bash
./run_keyword_reader.sh \
  /tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json \
  document_extracted
```

默认值：

- JSON 文件：`/tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json`
- 字段：`document_extracted`
