# 新老AI跨时空对话卡片系统

这个系统现在采用“Python 批量跑 + JSON 记录 + 后筛选”的方案。

原则：

> 话题卡可以设计刺激，但刺激必须对所有模型相同。运行时主持人只发卡，不临场加戏；旧AI暴露边界、提问、追问并交卷总结；新AI负责解释和回答。后筛选只标注和挑选，不改写原始回答。

## 为什么先用 Python，不做 PHP

当前阶段最重要的是批量、可复现、方便改卡片和 prompt。Python 脚本直接产出结构化 JSON，最接近林亦那类项目的工作方式，也最适合后面做批量实验。

后续如果需要更好的浏览体验，可以基于这些 JSON 再做一个轻量 HTML/Flask 查看器。PHP 后台暂时不必要。

## 三幕与七张卡

卡片配置在 [configs/cards.json](/Users/dahei/Documents/新老AI跨时空对话/configs/cards.json:1)。

第一幕：未来发生了什么

- `deepseek_r1`
- `black_myth_wukong`
- `xiaomi_su7`

第二幕：共同之物后来变了

- `memory_price_surge`
- `open_source_vs_closed_models`
- `ai_make_people_easier`

第三幕：告别

- `farewell_gpt4o`

## 运行协议

目前有三种协议：

- `unknown_event_exam_v1`：旧AI不知道或不完整知道的未来事件。新AI先问边界，旧AI提问，新AI回答，旧AI追问，旧AI交卷。
- `shared_concept_exam_v1`：双方都知道的共同概念，但2026年的理解发生变化。旧AI先说旧理解，新AI解释变化，旧AI追问并交卷。
- `farewell_exam_v1`：告别 GPT-4o。旧AI回应退场，新AI回应，旧AI提问，新AI回答，旧AI最后告别。

## 目录

```text
configs/
  models.json              模型配置，只需要填模型名、API key环境变量、base_url环境变量
  cards.json               七张话题卡
  run_plan.json            默认运行计划
prompts/
  old_ai_system.txt        旧AI“小白考官”提示词
  new_ai_system.txt        新AI解释者提示词
  editor_prompt.txt        后筛选导演评分提示词
records/
  raw/                     原始对话JSON
  selected/                后筛选片段JSON和Markdown
outputs/
  script_candidates/       脚本候选
  visual_cards/            画面卡片JSON
scripts/
  run_dialogue.py          只跑对话
  extract_highlights.py    只做后筛选
  build_script.py          只生成脚本候选
  run_cards_pipeline.py    对话 -> 后筛选 -> 脚本，一条命令跑完整链路
  run_mvp_pipeline.py      旧入口，保留兼容
```

## 安装

```bash
cd /Users/dahei/Documents/新老AI跨时空对话
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

然后编辑 `.env`：

```env
OPENAI_API_KEY=你的key
LLM_URL=你的OpenAI-compatible接口地址
```

如果不同模型走不同渠道，可以在 `configs/models.json` 里给每个模型单独设置 `api_key_env` 和 `base_url_env`，再在 `.env` 里填对应变量。

## 你需要配置什么

主要改 [configs/models.json](/Users/dahei/Documents/新老AI跨时空对话/configs/models.json:1)。

```json
{
  "old_early_gpt4o": {
    "display_name": "旧AI候选 早期GPT-4o",
    "model": "gpt-4o-2024-05-13",
    "role": "old",
    "cutoff_date": "2023-10",
    "api_key_env": "OPENAI_API_KEY",
    "base_url_env": "LLM_URL"
  },
  "current_gpt55": {
    "display_name": "GPT-5.5",
    "model": "gpt-5.5",
    "role": "new",
    "api_key_env": "OPENAI_API_KEY",
    "base_url_env": "LLM_URL"
  }
}
```

关键字段：

- `display_name`：报告里展示的名字。
- `model`：API调用时传给服务商的模型名。
- `role`：`old` / `new` / `editor`。
- `cutoff_date`：旧AI建议填写，用于系统提示里的时间边界。
- `api_key_env`：从哪个环境变量读 API key。
- `base_url_env`：从哪个环境变量读 OpenAI-compatible base URL。

## 不联网干跑

不填 API 也能验证链路：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --dry-run --limit-cards 1 --top-k 1
```

运行后会生成：

- `records/raw/*.json`
- `records/selected/*-highlights.json`
- `records/selected/*-highlights.md`
- `outputs/script_candidates/*-script.md`
- `outputs/visual_cards/*-visual-cards.json`

## 真实试跑

先跑一张卡：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --limit-cards 1 --top-k 1
```

跑指定卡：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --card black_myth_wukong --top-k 1
```

跑七张卡：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --top-k 7
```

同一张卡重复跑多次：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --card black_myth_wukong --repeat 3 --top-k 3
```

如果要按 `configs/run_plan.json` 的默认配置跑多个新模型：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --top-k 10
```

## 后筛选导演

默认后筛选使用本地启发式评分。

如果想让编辑器模型来判断“哪段更适合剪”，加 `--use-llm-editor`：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --use-llm-editor --top-k 7
```

后筛选评分包含：

- `hook_score`：可剪辑钩子
- `time_gap_score`：时间差信息
- `evaluation_value_score`：真实评测价值，如事实准确性、解释力、承认不确定性、追问质量
- `emotional_score`：自然出现的人味或复杂情绪
- `model_personality_score`：模型自然表达风格
- `factual_risk`：事实风险

注意：`fact_sheet` 只用于后筛选和人工核验，不会注入给模型。

## 修改卡片

编辑 [configs/cards.json](/Users/dahei/Documents/新老AI跨时空对话/configs/cards.json:1)。

每张卡的关键字段：

- `host_injection`：主持人发卡时说的话。
- `protocol`：使用哪种对话协议。
- `objective`：这张卡想观察什么。
- `evaluation_focus`：后筛选时关注哪些能力。
- `fact_sheet`：人工核验用，不给模型看。
- `tags`：后续检索和剪辑用。

## 推荐试跑顺序

1. 干跑一张卡，确认链路。
2. 真实跑 `black_myth_wukong` 一张卡，看旧AI提问和新AI回答质量。
3. 固定旧AI，跑一个新模型的七张卡。
4. 再扩到七个新模型。
5. 如果素材不够，再对单张高价值卡加 `--repeat 3`。

## 注意

涉及具体事实、日期、产品状态、价格、销量、模型版本的片段，输出里会标注 `factual_risk`。剪进正片前需要人工核验。
