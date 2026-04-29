from __future__ import annotations

import json
import re

from .config import AgentConfig
from .graph import GraphIndex
from .llm import LLMClient
from .models import Evidence, EvidencePack
from .store import KnowledgeStore
from .vector_store import search_chroma


class AgenticRetriever:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.store = KnowledgeStore(config.db_path)
        self.graph = GraphIndex(config.graph_jsons).load()

    def retrieve(self, query: str) -> EvidencePack:
        rewritten = self._rewrite_queries(query)
        all_queries = [query] + rewritten[: self.config.max_rewrites]
        seen_sources: set[tuple[str, str]] = set()
        evidence: list[Evidence] = []
        graph_terms = self.graph.expand_query_terms(query, limit=8)
        topics = self._infer_topics(query, graph_terms)
        graph_neighbors: list[str] = []
        for topic in topics[:3]:
            graph_neighbors.extend(self.graph.query_graph_neighbors(topic, limit=8))

        for current_query in all_queries:
            search_query = " ".join([current_query] + graph_terms[:5])
            rows = self.store.search(search_query, limit=max(self.config.top_k * 8, 40))
            if _looks_like_interview_query(current_query):
                rows.extend(
                    self.store.metadata_search(_query_terms(current_query), limit=80, source_prefix="01_notes/")
                )
            for row in rows:
                key = (row["source_file"], row["chunk_id"])
                if key in seen_sources:
                    continue
                seen_sources.add(key)
                snippet = self._make_snippet(row["text"], current_query)
                score = self._score(row["text"], current_query, topics, row["title"], row["source_file"])
                if score <= 0:
                    continue
                evidence.append(
                    Evidence(
                        title=row["title"],
                        source_file=row["source_file"],
                        source_url=row["source_url"] or "",
                        snippet=snippet,
                        score=score,
                        reason=self._reason(current_query, topics),
                    )
                )
            for item in search_chroma(self.config, search_query, limit=self.config.top_k):
                source_file = str(item.get("source_file") or "")
                chunk_id = str(item.get("chunk_id") or "")
                key = (source_file, chunk_id)
                if key in seen_sources:
                    continue
                seen_sources.add(key)
                text = str(item.get("text") or "")
                title = str(item.get("title") or "")
                score = self._score(text, current_query, topics, title, source_file)
                if score <= 0:
                    continue
                distance = float(item.get("distance") or 1.0)
                evidence.append(
                    Evidence(
                        title=title,
                        source_file=source_file,
                        source_url=str(item.get("source_url") or ""),
                        snippet=self._make_snippet(text, current_query),
                        score=score + max(0.0, 2.0 - distance),
                        reason=self._reason(current_query, topics) + "；Chroma向量召回",
                    )
                )
            evidence.sort(key=lambda item: item.score, reverse=True)

        return EvidencePack(
            query=query,
            rewritten_queries=rewritten,
            topics=topics,
            graph_neighbors=_dedupe(graph_neighbors)[:12],
            evidence=evidence[: self.config.top_k],
        )

    def answer(self, query: str) -> str:
        pack = self.retrieve(query)
        if not pack.evidence:
            return "没有在本地知识库中检索到足够证据。建议先补充相关笔记或换一个更具体的问题。"
        llm = LLMClient.from_config(self.config)
        lines = [f"问题：{query}", "", "回答："]
        lines.append(_synthesize_answer(query, pack, llm))
        lines.extend(["", "来源："])
        for idx, ev in enumerate(pack.evidence[:5], start=1):
            lines.append(f"{idx}. {ev.title} - {ev.source_file}")
            if ev.source_url:
                lines.append(f"   {ev.source_url}")
        return "\n".join(lines)

    def _rewrite_queries(self, query: str) -> list[str]:
        rewrites = []
        if "记忆" in query or "memory" in query.lower():
            rewrites.append("Agent Memory 分层记忆 短期 长期 技能 偏好")
            rewrites.append("Hermes Agent 记忆 Skills 用户偏好 任务记忆")
        if "RAG" in query.upper() or "检索" in query:
            rewrites.append("RAG GraphRAG Agentic RAG 检索 评估 召回 rerank")
            rewrites.append("JIT Context 上下文工程 RAG 逻辑探索")
        if "Hermes" in query or "OpenClaw" in query:
            rewrites.append("Hermes Agent OpenClaw skill 记忆 自我改进")
        if "项目" in query or "系统" in query:
            rewrites.append(f"{query} 架构 取舍 工程 落地 指标")
        return _dedupe(rewrites)

    def _infer_topics(self, query: str, graph_terms: list[str]) -> list[str]:
        topics = []
        for topic in ["Agent", "RAG", "Harness", "大模型", "Python", "工程开发", "面经", "求职实习", "算法", "产品运营"]:
            if topic.lower() in query.lower() or topic in graph_terms:
                topics.append(topic)
        if "记忆" in query or "memory" in query.lower():
            topics.append("Agent")
        if "Hermes" in query:
            topics.append("Agent")
        return _dedupe(topics or graph_terms[:3])

    def _make_snippet(self, text: str, query: str, max_len: int = 260) -> str:
        terms = sorted(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]+", query), key=len, reverse=True)
        pos = 0
        for term in terms:
            found = text.lower().find(term.lower())
            if found >= 0:
                pos = max(0, found - 60)
                break
        snippet = text[pos : pos + max_len].replace("\n", " ").strip()
        return snippet

    def _score(self, text: str, query: str, topics: list[str], title: str = "", source_file: str = "") -> float:
        lower = text.lower()
        title_lower = title.lower()
        source_lower = source_file.lower()
        haystack = f"{lower}\n{title_lower}\n{source_lower}"
        terms = _query_terms(query)
        score = sum(1.0 for term in terms if term.lower() in lower)
        score += sum(2.0 for term in terms if term.lower() in title_lower)
        score += sum(0.7 for topic in topics if topic.lower() in haystack)
        query_lower = query.lower()
        if "记忆" in query or "memory" in query_lower:
            score += sum(2.0 for term in ["记忆", "memory", "hermes", "skill", "偏好", "任务记忆"] if term in haystack)
        if "hermes" in query_lower or "openclaw" in query_lower:
            score += sum(2.0 for term in ["hermes", "openclaw", "skill", "自动", "失效"] if term in haystack)
        if "rag" in query_lower or "检索" in query:
            score += sum(1.5 for term in ["rag", "graphrag", "agentic", "检索", "评估", "召回"] if term in haystack)
        if _looks_like_interview_query(query) and source_file.startswith("01_notes/"):
            score += 4.0
            for company in ["字节", "阿里", "腾讯", "美团", "百度", "蚂蚁", "快手", "高德", "淘天", "小红书"]:
                if company in query and company in f"{title}{source_file}":
                    score += 8.0
            for round_name in ["一面", "二面", "三面"]:
                if round_name in query and round_name in title:
                    score += 3.0
        return score

    def _reason(self, query: str, topics: list[str]) -> str:
        if topics:
            return f"命中问题关键词，并关联主题：{', '.join(topics[:3])}"
        return "命中问题关键词"


def _synthesize_answer(query: str, pack: EvidencePack, llm: LLMClient | None = None) -> str:
    if llm and llm.enabled:
        generated = llm.complete(
            system=(
                "你是一个严格的面试学习 Agent。只能基于用户提供的证据回答；"
                "如果证据不足，要明确说明不足。回答要结构化、工程化，并保留可追溯引用编号。"
            ),
            user=(
                f"问题：{query}\n\n"
                f"证据：\n{_format_evidence_for_prompt(pack)}\n\n"
                "请输出：\n"
                "1. 直接结论\n"
                "2. 分点解释\n"
                "3. 工程取舍/评估指标/失败兜底\n"
                "4. 引用到的证据编号\n"
                "要求 600 字以内，但必须完整收尾。"
            ),
            max_output_tokens=1800,
        )
        if generated and not generated.startswith("[LLM_ERROR]"):
            return generated
        if generated.startswith("[LLM_ERROR]"):
            return generated + "\n\n" + _offline_synthesize_answer(query, pack)
    return _offline_synthesize_answer(query, pack)


def _offline_synthesize_answer(query: str, pack: EvidencePack) -> str:
    snippets = " ".join(ev.snippet for ev in pack.evidence[:4])
    points: list[str] = []
    if "记忆" in query or "memory" in query.lower():
        points = [
            "面试学习 Agent 的记忆应做分层：短期保存当前问答和追问链，中期保存会话摘要，长期保存薄弱主题、用户偏好和题目掌握度。",
            "Hermes 风格的关键不是把所有历史都塞进上下文，而是只保存重要信息，并把重复成功/失败流程沉淀为可路由的 Skills。",
            "检索时把长期记忆映射回知识图谱主题，让下一轮训练优先召回薄弱点相关资料。",
        ]
    elif "RAG" in query.upper() or "检索" in query:
        points = [
            "传统 RAG 适合语义相似问题，但复杂工程题需要 Agentic RAG：检索后评估证据是否足够，不足时重写查询并继续检索。",
            "GraphRAG 用主题、标签、笔记关系补足向量 Top-K 的盲区，适合面试系统设计和跨文档归纳题。",
            "评估应同时看检索召回、证据相关性、生成忠实度、引用覆盖率和端到端稳定性。",
        ]
    elif "Hermes" in query or "OpenClaw" in query:
        points = [
            "Hermes 的核心差异在于自动沉淀技能、自我修复失效步骤、分层管理记忆，而不是单纯依赖手写技能文档。",
            "OpenClaw 更偏人工编写和维护 skill；Hermes 更强调从执行经验中学习，把任务流程和偏好沉淀下来。",
            "工程实现上应把候选技能先放入 pending，经过人工确认后再进入 active，避免错误经验污染系统。",
        ]
    else:
        points = [
            "先明确题型和目标，再用关键词、向量/全文、图谱三路召回证据。",
            "回答必须围绕证据组织，并补充工程取舍、失败模式和评估指标。",
            "复盘时把薄弱点写回长期记忆，形成下一轮训练闭环。",
        ]
    return "\n".join(f"- {point}" for point in points) + f"\n\n证据摘要：{snippets[:420]}"


def _format_evidence_for_prompt(pack: EvidencePack, limit: int = 6) -> str:
    lines = []
    for idx, ev in enumerate(pack.evidence[:limit], start=1):
        lines.append(
            f"[{idx}] title={ev.title}\n"
            f"source_file={ev.source_file}\n"
            f"source_url={ev.source_url}\n"
            f"snippet={ev.snippet}\n"
        )
    return "\n".join(lines)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _query_terms(query: str) -> list[str]:
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query)
    terms = chinese + latin
    for token in chinese:
        if len(token) > 3:
            terms.extend(token[i : i + 2] for i in range(0, len(token) - 1))
    return _dedupe(terms)


def _looks_like_interview_query(query: str) -> bool:
    markers = ["面经", "一面", "二面", "三面", "面试", "实习", "校招", "问了什么"]
    companies = ["字节", "阿里", "腾讯", "美团", "百度", "蚂蚁", "快手", "高德", "淘天", "小红书"]
    return any(marker in query for marker in markers) or any(company in query for company in companies)
