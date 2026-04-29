from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import AgentConfig
from .ingest import ensure_workspace
from .memory import MemoryManager, knowledge_points_for_topic
from .sessions import SessionLog, new_session_id
from .workflow import GraphState, prepare_interview_question, run_interactive_turn


PENDING_QUESTIONS: dict[str, GraphState] = {}


def run_web_app(config: AgentConfig, host: str = "127.0.0.1", port: int = 8765) -> None:
    ensure_workspace(config)

    class InterviewHandler(_InterviewHandler):
        agent_config = config

    server = ThreadingHTTPServer((host, port), InterviewHandler)
    print(f"Interview web UI: http://{host}:{port}")
    server.serve_forever()


class _InterviewHandler(BaseHTTPRequestHandler):
    agent_config: AgentConfig

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            if self.path == "/api/session":
                self._send_json(self._start_session(body))
            elif self.path == "/api/question":
                self._send_json(self._next_question(body))
            elif self.path == "/api/answer":
                self._send_json(self._submit_answer(body))
            elif self.path == "/api/skip":
                self._send_json(self._skip_question(body))
            elif self.path == "/api/progress":
                self._send_json(self._progress(body))
            elif self.path == "/api/gaps":
                self._send_json(self._gaps(body))
            else:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _start_session(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        rounds = max(1, min(20, int(body.get("rounds") or 5)))
        difficulty = str(body.get("difficulty") or "medium").strip() or "medium"
        knowledge_point = str(body.get("knowledge_point") or "auto").strip() or "auto"
        review_first = bool(body.get("review_first", True))
        session_id = str(body.get("session_id") or "").strip() or new_session_id()
        log = SessionLog(self.agent_config, session_id)
        turns = _turn_events(log)
        if not turns:
            log.append(
                "session_start",
                {
                    "topic": topic,
                    "rounds": rounds,
                    "mode": "web",
                    "difficulty": difficulty,
                    "knowledge_point": knowledge_point,
                    "review_first": review_first,
                },
            )
        return {
            "session_id": session_id,
            "topic": topic,
            "rounds": rounds,
            "difficulty": difficulty,
            "knowledge_point": knowledge_point,
            "review_first": review_first,
            "completed_rounds": len(turns),
            "complete": len(turns) >= rounds,
            "knowledge_points": knowledge_points_for_topic(topic),
        }

    def _next_question(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        rounds = max(1, min(20, int(body.get("rounds") or 5)))
        difficulty = str(body.get("difficulty") or "medium").strip() or "medium"
        knowledge_point = str(body.get("knowledge_point") or "auto").strip() or "auto"
        review_first = bool(body.get("review_first", True))
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        log = SessionLog(self.agent_config, session_id)
        round_index = len(_turn_events(log)) + len(_skip_events(log))
        if round_index >= rounds:
            return {"complete": True, "round": round_index, "rounds": rounds}
        used_points = [
            str(event["payload"].get("knowledge_point", ""))
            for event in _turn_events(log) + _skip_events(log)
            if event["payload"].get("knowledge_point")
        ]
        exclude_points = used_points if knowledge_point == "auto" else []
        state = prepare_interview_question(
            self.agent_config,
            topic,
            round_index,
            difficulty,
            knowledge_point,
            review_first,
            exclude_points,
        )
        PENDING_QUESTIONS[session_id] = state
        evidence = (state.get("evidence_pack") or {}).get("evidence", [])
        return {
            "complete": False,
            "session_id": session_id,
            "round": round_index + 1,
            "rounds": rounds,
            "question": state["question"],
            "knowledge_point": state.get("knowledge_point", ""),
            "question_type": state.get("question_type", ""),
            "difficulty": state.get("difficulty", difficulty),
            "is_review": state.get("is_review", False),
            "evidence": [
                {
                    "title": item.get("title", ""),
                    "source_file": item.get("source_file", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in evidence[:4]
            ],
        }

    def _submit_answer(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        rounds = max(1, min(20, int(body.get("rounds") or 5)))
        session_id = str(body.get("session_id") or "").strip()
        answer = str(body.get("answer") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        if not answer:
            raise ValueError("answer is required")
        question_state = PENDING_QUESTIONS.get(session_id)
        if not question_state:
            raise ValueError("no pending question; request /api/question first")

        log = SessionLog(self.agent_config, session_id)
        round_index = len(_turn_events(log))
        result = run_interactive_turn(self.agent_config, session_id, topic, round_index, question_state, answer)
        turns = _turn_events(log)
        questions = [str(event["payload"].get("question", "")) for event in turns]
        answers = [str(event["payload"].get("answer", "")) for event in turns]
        questions.append(str(result["question"]))
        answers.append(answer)
        working_summary = MemoryManager(self.agent_config).build_working_summary(questions, answers)
        log.append(
            "turn",
            {
                "round": round_index + 1,
                "question": result["question"],
                "answer": answer,
                "evidence": result["evidence_pack"],
                "selected_skills": result.get("selected_skills", []),
                "knowledge_point": result.get("knowledge_point", ""),
                "question_type": result.get("question_type", ""),
                "difficulty": result.get("difficulty", ""),
                "is_review": result.get("is_review", False),
                "report": result["report"],
                "created_skills": result.get("created_skills", []),
                "memory_update": result.get("memory_update", {}),
                "working_summary": working_summary,
            },
        )
        complete = round_index + 1 >= rounds
        if complete:
            log.append("session_end", {"topic": topic, "rounds": rounds, "mode": "web"})
            PENDING_QUESTIONS.pop(session_id, None)
        return {
            "session_id": session_id,
            "round": round_index + 1,
            "rounds": rounds,
            "complete": complete,
            "question": result["question"],
            "knowledge_point": result.get("knowledge_point", ""),
            "question_type": result.get("question_type", ""),
            "difficulty": result.get("difficulty", ""),
            "is_review": result.get("is_review", False),
            "report": result["report"],
            "created_skills": result.get("created_skills", []),
            "memory_update": result.get("memory_update", {}),
            "progress": MemoryManager(self.agent_config).get_progress_summary(topic),
            "gaps": MemoryManager(self.agent_config).get_learning_gaps(topic),
            "session_file": str(log.path),
        }

    def _skip_question(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        rounds = max(1, min(20, int(body.get("rounds") or 5)))
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        question_state = PENDING_QUESTIONS.get(session_id)
        if not question_state:
            raise ValueError("no pending question; request /api/question first")
        log = SessionLog(self.agent_config, session_id)
        round_index = len(_turn_events(log)) + len(_skip_events(log))
        log.append(
            "skip",
            {
                "round": round_index + 1,
                "question": question_state.get("question", ""),
                "knowledge_point": question_state.get("knowledge_point", ""),
                "question_type": question_state.get("question_type", ""),
                "difficulty": question_state.get("difficulty", ""),
                "is_review": question_state.get("is_review", False),
            },
        )
        PENDING_QUESTIONS.pop(session_id, None)
        complete = round_index + 1 >= rounds
        if complete:
            log.append("session_end", {"topic": topic, "rounds": rounds, "mode": "web"})
        return {
            "session_id": session_id,
            "round": round_index + 1,
            "rounds": rounds,
            "complete": complete,
            "skipped": True,
        }

    def _progress(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        data = MemoryManager(self.agent_config).get_progress_summary(topic)
        data["knowledge_points"] = knowledge_points_for_topic(topic)
        return data

    def _gaps(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        return MemoryManager(self.agent_config).get_learning_gaps(topic)

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _turn_events(log: SessionLog) -> list[dict[str, Any]]:
    return [event for event in log.read_events() if event.get("event") == "turn"]


def _skip_events(log: SessionLog) -> list[dict[str, Any]]:
    return [event for event in log.read_events() if event.get("event") == "skip"]


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Interview Agent</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #65717d;
      --border: #d7dce0;
      --accent: #176b87;
      --accent-dark: #0f5268;
      --warn: #9a4b12;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, "Microsoft YaHei", sans-serif;
      line-height: 1.45;
    }
    header {
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: calc(100vh - 57px);
    }
    aside {
      border-right: 1px solid var(--border);
      background: var(--panel);
      padding: 18px;
    }
    section { padding: 22px; }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 14px 0 6px;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fff;
    }
    textarea {
      min-height: 180px;
      resize: vertical;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
    }
    button.secondary {
      color: var(--accent);
      background: #fff;
    }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .meta { color: var(--muted); font-size: 13px; }
    .question {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      font-size: 17px;
      margin-bottom: 16px;
    }
    .output {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
      white-space: pre-wrap;
    }
    .evidence {
      margin-top: 14px;
      display: grid;
      gap: 10px;
    }
    .source {
      border-left: 3px solid var(--accent);
      background: #f9fbfc;
      padding: 9px 11px;
      font-size: 13px;
    }
    .score {
      color: var(--accent-dark);
      font-weight: 700;
      margin-bottom: 8px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      margin-top: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }
    .kp {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .bar {
      height: 7px;
      background: #e8ecef;
      border-radius: 99px;
      overflow: hidden;
      margin: 8px 0;
    }
    .fill { height: 100%; background: var(--accent); }
    .status-unseen .fill { background: #a9b1b8; }
    .status-weak .fill { background: #b85042; }
    .status-improving .fill { background: #c18f1a; }
    .status-mastered .fill { background: #2d7d46; }
    .error { color: var(--warn); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <header>
    <h1>RAG Interview Agent</h1>
    <div class="meta" id="sessionMeta">No session</div>
  </header>
  <main>
    <aside>
      <label for="topic">Topic</label>
      <input id="topic" value="RAG">
      <label for="rounds">Rounds</label>
      <input id="rounds" type="number" value="5" min="1" max="20">
      <label for="difficulty">Difficulty</label>
      <select id="difficulty">
        <option value="easy">easy</option>
        <option value="medium" selected>medium</option>
        <option value="hard">hard</option>
      </select>
      <label for="knowledgePoint">Knowledge point</label>
      <select id="knowledgePoint"><option value="auto">auto</option></select>
      <label class="row" style="margin-top:14px">
        <input id="reviewFirst" type="checkbox" checked style="width:auto">
        <span>优先复习到期题</span>
      </label>
      <label for="session">Session ID</label>
      <input id="session" placeholder="leave empty for new session">
      <div class="row" style="margin-top:16px">
        <button id="startBtn">Start</button>
        <button id="nextBtn" class="secondary" disabled>Next</button>
      </div>
      <div class="meta" style="margin-top:14px" id="status">Ready</div>
    </aside>
    <section>
      <div class="question" id="question">Start a session to get the first question.</div>
      <textarea id="answer" placeholder="Type your answer here..."></textarea>
      <div class="row" style="margin-top:12px">
        <button id="submitBtn" disabled>Submit Answer</button>
        <button id="skipBtn" class="secondary" disabled>Skip</button>
      </div>
      <div class="panel">
        <strong>知识点掌握</strong>
        <div class="grid" id="mastery"></div>
      </div>
      <div class="panel">
        <strong>补学建议</strong>
        <div id="gaps" class="meta">完成一道题后显示补学建议。</div>
      </div>
      <div class="evidence" id="evidence"></div>
      <div class="output" id="report">Review will appear here.</div>
    </section>
  </main>
  <script>
    let sessionId = "";
    let topic = "RAG";
    let rounds = 5;
    let difficulty = "medium";
    let knowledgePoint = "auto";
    const $ = id => document.getElementById(id);

    function setBusy(isBusy, text) {
      $("status").textContent = text || (isBusy ? "Working..." : "Ready");
      $("startBtn").disabled = isBusy;
      $("nextBtn").disabled = isBusy || !sessionId;
      $("submitBtn").disabled = isBusy || !sessionId;
      $("skipBtn").disabled = isBusy || !sessionId;
    }

    async function post(path, body) {
      const res = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }

    function renderEvidence(items) {
      $("evidence").innerHTML = "";
      for (const item of items || []) {
        const div = document.createElement("div");
        div.className = "source";
        div.textContent = `${item.title} - ${item.source_file}\n${item.snippet || ""}`;
        $("evidence").appendChild(div);
      }
    }

    function renderReport(data) {
      const r = data.report || {};
      const scores = ["correctness","structure","engineering_depth","tradeoff_quality","source_grounding","anti_followup"]
        .map(k => Number(r[k] || 0));
      const avg = scores.length ? (scores.reduce((a,b) => a + b, 0) / scores.length).toFixed(2) : "0.00";
      const missing = (r.missing_points || []).map(x => `- ${x}`).join("\n");
      const tasks = (r.next_tasks || []).map(x => `- ${x}`).join("\n");
      $("report").innerHTML =
        `<div class="score">Score: ${avg}/5 · ${escapeText(data.knowledge_point || "")} · ${escapeText(data.difficulty || "")}</div>` +
        `<div>${escapeText("Missing points:\n" + (missing || "- none") + "\n\nBetter answer:\n" + (r.better_answer || "") + "\n\nNext tasks:\n" + (tasks || "- none"))}</div>`;
    }

    function renderKnowledgeOptions(items) {
      const current = $("knowledgePoint").value || "auto";
      $("knowledgePoint").innerHTML = `<option value="auto">auto</option>`;
      for (const item of items || []) {
        const opt = document.createElement("option");
        opt.value = item;
        opt.textContent = item;
        $("knowledgePoint").appendChild(opt);
      }
      $("knowledgePoint").value = [...$("knowledgePoint").options].some(o => o.value === current) ? current : "auto";
    }

    function renderProgress(data) {
      const topicData = data.topics && data.topics[topic];
      const points = topicData ? topicData.knowledge_points : [];
      renderKnowledgeOptions(data.knowledge_points || points.map(x => x.knowledge_point));
      $("mastery").innerHTML = "";
      for (const item of points || []) {
        const div = document.createElement("div");
        div.className = `kp status-${item.status}`;
        const pct = Math.max(0, Math.min(100, Number(item.recent_score || 0) * 20));
        div.innerHTML = `<strong>${escapeText(item.knowledge_point)}</strong>
          <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
          <div class="meta">${escapeText(item.status)} · attempts ${item.attempts} · recent ${item.recent_score || 0}</div>
          <div class="meta">next due: ${escapeText(item.next_due_at || "-")}</div>`;
        $("mastery").appendChild(div);
      }
    }

    function renderGaps(data) {
      const lines = (data.items || []).map(item =>
        `• ${item.knowledge_point}: ${(item.suggestions || []).slice(0, 4).join(" / ")}\n  next: ${item.next_practice || ""}`
      );
      $("gaps").textContent = lines.join("\n\n") || "No suggestions yet.";
    }

    async function refreshProgress() {
      const progress = await post("/api/progress", {topic});
      renderProgress(progress);
    }

    function escapeText(text) {
      return String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
    }

    async function startSession() {
      try {
        setBusy(true, "Creating session...");
        topic = $("topic").value.trim() || "RAG";
        rounds = Number($("rounds").value || 5);
        difficulty = $("difficulty").value;
        knowledgePoint = $("knowledgePoint").value || "auto";
        const data = await post("/api/session", {
          topic,
          rounds,
          difficulty,
          knowledge_point: knowledgePoint,
          review_first: $("reviewFirst").checked,
          session_id: $("session").value.trim()
        });
        sessionId = data.session_id;
        $("session").value = sessionId;
        $("sessionMeta").textContent = `${sessionId} · ${data.completed_rounds}/${data.rounds}`;
        await refreshProgress();
        await nextQuestion();
      } catch (err) {
        $("status").innerHTML = `<span class="error">${escapeText(err.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    async function nextQuestion() {
      try {
        setBusy(true, "Generating question...");
        difficulty = $("difficulty").value;
        knowledgePoint = $("knowledgePoint").value || "auto";
        const data = await post("/api/question", {
          session_id: sessionId,
          topic,
          rounds,
          difficulty,
          knowledge_point: knowledgePoint,
          review_first: $("reviewFirst").checked
        });
        if (data.complete) {
          $("question").textContent = "Session complete.";
          $("submitBtn").disabled = true;
          return;
        }
        const marker = data.is_review ? "review" : "new";
        $("question").textContent = `Round ${data.round}/${data.rounds} · ${data.knowledge_point} · ${data.difficulty} · ${marker}\n${data.question}`;
        $("answer").value = "";
        $("report").textContent = "Review will appear here.";
        $("gaps").textContent = "完成本道题后显示补学建议。";
        $("sessionMeta").textContent = `${sessionId} · ${data.round - 1}/${data.rounds}`;
        renderEvidence(data.evidence);
      } catch (err) {
        $("status").innerHTML = `<span class="error">${escapeText(err.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    async function submitAnswer() {
      try {
        const answer = $("answer").value.trim();
        if (!answer) return;
        setBusy(true, "Scoring answer...");
        const data = await post("/api/answer", {session_id: sessionId, topic, rounds, answer});
        renderReport(data);
        if (data.progress) renderProgress(data.progress);
        if (data.gaps) renderGaps(data.gaps);
        $("sessionMeta").textContent = `${sessionId} · ${data.round}/${data.rounds}`;
        $("status").textContent = data.complete ? `Complete. Saved: ${data.session_file}` : "Scored. Click Next.";
      } catch (err) {
        $("status").innerHTML = `<span class="error">${escapeText(err.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    async function skipQuestion() {
      try {
        if (!sessionId) return;
        setBusy(true, "Skipping question...");
        const data = await post("/api/skip", {session_id: sessionId, topic, rounds});
        $("answer").value = "";
        $("report").textContent = "Skipped. This answer was not written to long-term memory.";
        $("gaps").textContent = "跳过题目不会生成补学建议。";
        $("sessionMeta").textContent = `${sessionId} · ${data.round}/${data.rounds}`;
        if (data.complete) {
          $("question").textContent = "Session complete.";
          $("status").textContent = "Complete.";
        } else {
          $("status").textContent = "Skipped. Click Next.";
        }
      } catch (err) {
        $("status").innerHTML = `<span class="error">${escapeText(err.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    $("startBtn").addEventListener("click", startSession);
    $("nextBtn").addEventListener("click", nextQuestion);
    $("submitBtn").addEventListener("click", submitAnswer);
    $("skipBtn").addEventListener("click", skipQuestion);
    $("topic").addEventListener("change", async () => {
      topic = $("topic").value.trim() || "RAG";
      try { await refreshProgress(); } catch (err) {}
    });
    refreshProgress().catch(() => {});
  </script>
</body>
</html>
"""
