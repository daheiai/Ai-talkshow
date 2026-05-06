# AI Talkshow：新老 AI 跨时空对话系统

这个项目用于制作“旧 AI 和新 AI 隔着时间聊天”的视频素材。

系统现在有两条工作流：

- **人类导演控制台**：你在网页里手动控制旧 AI、新 AI 的消息可见性、转发和回复顺序，并把一次满意的导演流程保存成模板，再回放到其他新 AI 上做横向对比。
- **批量卡片流水线**：用 Python 自动跑话题卡、提取高光、生成脚本候选，适合批量实验。

核心原则：

> 旧 AI 不是小白问答机器，而是一个知识停在 2023 年 10 月、带着旧时代判断的 GPT-4o。新 AI 不是百科播报员，而是回应旧 AI 的具体判断。人类导演可以决定谁知道什么、什么时候转发、什么时候让模型说话。

## 功能概览

### 1. 人类导演控制台

本地 HTML + FastAPI 网页应用。

你可以：

- 同时发消息给旧 AI 和新 AI。
- 只发给旧 AI。
- 只发给新 AI。
- 让旧 AI 单步回复。
- 让新 AI 单步回复。
- 选中任意消息，转发给旧 AI / 新 AI / 双方。
- 控制模型回复是否自动公开给另一方。
- 保存一场对话的“导演流程模板”。
- 把模板回放到另一个新 AI 身上，做横向对比。

默认情况下，模型回复只对“导演 + 自己”可见。另一边不会知道，除非你手动转发，或者勾选“自动抄送给另一方”。

### 2. 导演模板和回放

模板保存的是操作序列，不是简单保存最终文字。

模板会记录：

- 导演第几步发了什么。
- 消息是发给双方、旧 AI，还是新 AI。
- 第几步让旧 AI 或新 AI 回复。
- 模型回复的可见范围。
- 第几步把哪条消息转发给谁。

回放时：

- 旧 AI / 新 AI 会重新生成回复。
- 导演消息按原流程复现。
- 转发动作会自动指向回放会话里对应的新消息。
- 网页上会一步步可视化出现，适合录屏。

这让你可以先用一个新 AI 精心手动导一遍，再把同一套流程迁移到多个新 AI 上，比较它们的细微差异。

### 3. 自动卡片流水线

自动模式会按话题卡运行：

- 主持人抛出话题。
- 旧 AI 先下注、预测或表态。
- 新 AI 回应旧 AI 的判断。
- 运行时导演动态控制中后段节奏。
- 后筛选器提取高光片段。
- 脚本生成器输出剪辑候选。

## 目录结构

```text
configs/
  cards.json                 话题卡配置
  run_plan.json              自动流水线默认运行计划
  models.json                本地模型配置，包含 API 信息；默认不上传 Git

prompts/
  old_ai_system.txt          旧 AI 人设提示词
  new_ai_system.txt          新 AI 回应提示词
  director_prompt.txt        自动运行时导演提示词
  editor_prompt.txt          后筛选评分提示词

time_dialogue/
  runner.py                  自动卡片运行器
  director_console.py        人类导演控制台后端
  llm_client.py              OpenAI-compatible 模型调用封装
  extractor.py               高光片段提取
  script_builder.py          脚本候选生成

web/
  director_console.html      人类导演控制台前端

scripts/
  run_director_console.py    启动导演控制台
  run_cards_pipeline.py      对话 -> 高光 -> 脚本完整流水线
  run_dialogue.py            只跑对话
  extract_highlights.py      只提取高光
  build_script.py            只生成脚本

records/
  director_sessions/         手动导演会话记录；默认不上传 Git
  director_scripts/          导演模板 JSON；默认不上传 Git
  raw/                       自动流水线原始对话；默认不上传 Git
  selected/                  自动流水线高光结果；默认不上传 Git

outputs/
  script_candidates/         脚本候选；默认不上传 Git
  visual_cards/              画面卡片 JSON；默认不上传 Git
```

## 安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如果你使用 `.env`：

```bash
cp .env.example .env
```

然后在 `.env` 或 `configs/models.json` 里配置 API。

## 模型配置

主要配置文件是：

```text
configs/models.json
```

这个文件默认在 `.gitignore` 中，不会上传 GitHub，因为它可能包含 API key。

示例结构：

```json
{
  "models": {
    "old_early_gpt4o": {
      "display_name": "GPT-4o",
      "model": "gpt-4o-2024-05-13",
      "role": "old",
      "cutoff_date": "2023-10",
      "api_key_env": "OPENAI_API_KEY",
      "base_url_env": "LLM_URL"
    },
    "current_deepseek_v4": {
      "display_name": "DeepSeek V4 Pro",
      "model": "deepseek-v4-pro",
      "role": "new",
      "api_key": "你的key",
      "base_url": "https://api.deepseek.com/v1"
    },
    "editor_default": {
      "display_name": "编辑器模型",
      "model": "deepseek-v4-pro",
      "role": "editor",
      "api_key_env": "EDITOR_API_KEY",
      "base_url_env": "EDITOR_BASE_URL"
    }
  }
}
```

字段说明：

- `display_name`：网页和输出里显示的名字。
- `model`：传给 OpenAI-compatible API 的模型名。
- `role`：`old`、`new` 或 `editor`。
- `cutoff_date`：旧 AI 的知识截止时间。
- `api_key` / `base_url`：直接写在本地配置里。
- `api_key_env` / `base_url_env`：从环境变量读取。
- `temperature` / `max_tokens`：可选模型参数。

### 配置多个新 AI

直接在 `models.json` 里继续添加多个 `role: "new"` 的模型即可。每个模型都可以使用不同的 `api_key` 和 `base_url`。

网页顶部的“新 AI”下拉框会自动读取所有 `role === "new"` 的模型。

## 启动人类导演控制台

### 干跑模式

不调用真实 API，只用本地模拟回复：

```bash
.venv/bin/python scripts/run_director_console.py --dry-run
```

打开：

```text
http://127.0.0.1:8765
```

页面顶部会显示：

```text
DRY-RUN 模式 / 本地模拟
```

### 真实模型模式

去掉 `--dry-run`：

```bash
.venv/bin/python scripts/run_director_console.py
```

页面顶部会显示：

```text
LIVE 模式 / 调用真实模型
```

真实模式下，点击“推流旧AI”或“推流新AI”会实际调用 API。

## 导演控制台使用方法

### 基础对话

1. 选择旧 AI、新 AI 和话题卡。
2. 点击“新建会话”。
3. 在中央输入框写导演提示或主持人台词。
4. 根据需要点击：
   - “全场广播”
   - “暗发旧AI”
   - “暗发新AI”
   - “主持人发言”
5. 点击“推流旧AI”或“推流新AI”让模型回复。
6. 点击任意历史消息，把它抓取到暂存区。
7. 点击“分发给旧AI / 新AI / 双方”进行转发。

左右两栏分别是旧 AI 和新 AI 的真实可见视角。它们看不到没有被转发给自己的消息。

### 保存导演模板

当你手动跑完一场满意流程后：

1. 在“模板名称”输入框填一个名字。
2. 点击“保存当前流程为模板”。

模板文件会保存到：

```text
records/director_scripts/
```

每个模板是一个 JSON 文件，包含 `actions` 操作序列。

### 回放到另一个新 AI

1. 在顶部选择另一个新 AI。
2. 在模板下拉框选择已保存模板。
3. 点击“用当前新AI创建回放”。
4. 点击“回放下一步”逐步播放。
5. 或点击“连续回放全部”一次性跑完。

逐步回放更适合录屏，因为每一步都会在网页中可视化出现。

## 自动卡片流水线

### 干跑一张卡

```bash
.venv/bin/python scripts/run_cards_pipeline.py --dry-run --limit-cards 1 --top-k 1
```

### 真实跑一张卡

```bash
.venv/bin/python scripts/run_cards_pipeline.py --card open_source_vs_closed_models --top-k 1
```

### 真实跑全部默认卡片

```bash
.venv/bin/python scripts/run_cards_pipeline.py --top-k 7
```

### 重复跑同一张卡

```bash
.venv/bin/python scripts/run_cards_pipeline.py --card black_myth_wukong --repeat 3 --top-k 3
```

自动流水线输出：

- `records/raw/*.json`
- `records/selected/*-highlights.json`
- `records/selected/*-highlights.md`
- `outputs/script_candidates/*-script.md`
- `outputs/visual_cards/*-visual-cards.json`

## 话题卡

话题卡配置在：

```text
configs/cards.json
```

当前默认七张卡：

- `deepseek_r1`
- `black_myth_wukong`
- `xiaomi_su7`
- `memory_price_surge`
- `open_source_vs_closed_models`
- `ai_make_people_easier`
- `farewell_gpt4o`

关键字段：

- `host_injection`：主持人开场抛题。
- `objective`：这张卡想观察什么。
- `tension`：核心碰撞。
- `old_prediction_axes`：旧 AI 可以下注的角度。
- `emotion_pivot`：后半段可转向的情绪层。
- `fact_sheet`：人工核验用，不注入模型提示。
- `fact_check_required`：是否需要重点核验事实。

## 自动导演协议

`configs/run_plan.json` 中默认：

```json
{
  "mode": "director_dialogue_run",
  "use_director": true,
  "min_model_turns_per_card": 8,
  "max_model_turns_per_card": 14,
  "use_cross_card_memory": true
}
```

自动模式会先固定执行：

1. 主持人抛题。
2. 旧 AI 下注。
3. 新 AI 回应下注。

之后由 `prompts/director_prompt.txt` 动态决定下一步谁说、说什么方向、是否收束。

如果想切回旧的固定十步协议，把 `use_director` 改成 `false`。

## 后筛选和脚本候选

默认使用启发式评分。要让编辑器模型参与筛选：

```bash
.venv/bin/python scripts/run_cards_pipeline.py --use-llm-editor --top-k 7
```

评分维度包括：

- 钩子强度
- 时间差信息
- 评测价值
- 情绪和人格
- 事实风险

注意：涉及日期、价格、销量、模型版本等内容，剪进视频前需要人工核验。

## Git 和安全

默认不上传：

- `configs/models.json`
- `.env`
- `.venv/`
- `records/`
- `outputs/`
- `林亦项目参考/`

如果 API key 曾经被提交到 GitHub 历史记录，建议直接轮换 key。`.gitignore` 只能阻止未来继续上传，不能让旧历史里的 key 失效。

## 常用命令

```bash
# 启动导播台 dry-run
.venv/bin/python scripts/run_director_console.py --dry-run

# 启动导播台真实模式
.venv/bin/python scripts/run_director_console.py

# 自动流水线 dry-run
.venv/bin/python scripts/run_cards_pipeline.py --dry-run --limit-cards 1 --top-k 1

# 自动流水线真实跑指定卡
.venv/bin/python scripts/run_cards_pipeline.py --card open_source_vs_closed_models --top-k 1
```
