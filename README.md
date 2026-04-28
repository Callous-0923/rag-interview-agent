# 面试学习 Agent

本项目是一个本地 CLI + 文件存储的面试学习 Agent，读取上层知识库中的小红书笔记、索引和 Graphify 图谱，实现 Agentic RAG、模拟面试、复盘记忆和 Hermes 风格技能沉淀。

## 快速开始

```powershell
cd D:\files\knowledge\04_Interview_Agent
python -m interview_agent.cli init
python -m interview_agent.cli vision ingest --limit 5 --update-index
python -m interview_agent.cli ingest
python -m interview_agent.cli ask "Agent记忆系统怎么设计"
python -m interview_agent.cli web
python -m interview_agent.cli interview --topic RAG --rounds 5
python -m interview_agent.cli mock --topic RAG --rounds 3
python -m interview_agent.cli review --session <session_id>
python -m interview_agent.cli skills list
```

当前实现默认使用 SQLite FTS/关键词检索，并在安装了 ChromaDB 时同步写入 `storage\chroma`；没有 ChromaDB 时不影响主闭环。

## 交互式模拟面试

`interview` 是真人练习入口。系统一次只问一个问题，等待你输入答案，空行结束本轮回答，然后自动评分、复盘并写入 session。

```powershell
python -m interview_agent.cli interview --topic RAG --rounds 5
```

可用命令：

- `:hint`：查看本题检索到的证据提示
- `:skip`：跳过当前问题
- `:quit`：暂停 session

继续同一个 session：

```powershell
python -m interview_agent.cli interview --topic RAG --rounds 5 --session <session_id>
```

`mock` 保留为自动示范模式：系统自己生成问题、示范回答、评分和复盘，适合批量生成训练样例。

## Web 页面

启动本地前端页面：

```powershell
python -m interview_agent.cli web
```

打开：

```text
http://127.0.0.1:8765
```

页面支持创建/继续 session、生成下一题、提交回答、查看证据提示和评分复盘。后端仍然复用 CLI 的检索、出题、评分、记忆和技能沉淀逻辑。

## 视觉入库

小红书图片不会直接在问答时反复送入模型，而是先转成可检索 Markdown：

```powershell
python -m interview_agent.cli vision ingest --limit 20 --update-index
```

- 输出目录：`vision\xhs_image_notes`
- 缓存目录：`vision\cache`
- token 日志：`vision\usage.jsonl`
- `--dry-run` 只统计待处理图片
- `--force` 会重新调用视觉模型覆盖已有结果
- `--update-index` 会在生成 Markdown 后重建 SQLite/Chroma 索引

可在 `config.yaml` 中关闭图片处理：

```yaml
vision:
  enabled: false
```

重建小红书 Graphify 图谱时，`..\_scripts\build_xhs_graphify_graph.py` 会自动读取这些视觉 Markdown，将图片节点连接到 `vision_text` 节点。

## 数据源

默认只读读取：

- `..\01_XHS_Notes\notes\*.md`
- `.\vision\xhs_image_notes\*.md`
- `..\00_Inbox\实习面经\01_notes\**\*.md`
- `..\01_XHS_Notes\indexes\topic_index.md`
- `..\01_XHS_Notes\indexes\topic_assignments.csv`
- `..\01_XHS_Notes\graphify-out\graph.json`
- `..\00_Inbox\实习面经\graphify-out\graph.json`

## 输出

- `sessions\*.jsonl`：模拟面试与问答轨迹
- `memory\weakness_map.md`：长期短板记忆
- `memory\topic_mastery.json`：主题掌握度
- `skills\pending\*\SKILL.md`：候选技能
- `skills\active\*\SKILL.md`：已启用技能
