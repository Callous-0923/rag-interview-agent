from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import AgentConfig
from .ingest import ensure_workspace
from .memory import MemoryManager
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
        session_id = str(body.get("session_id") or "").strip() or new_session_id()
        log = SessionLog(self.agent_config, session_id)
        turns = _turn_events(log)
        if not turns:
            log.append("session_start", {"topic": topic, "rounds": rounds, "mode": "web"})
        return {
            "session_id": session_id,
            "topic": topic,
            "rounds": rounds,
            "completed_rounds": len(turns),
            "complete": len(turns) >= rounds,
        }

    def _next_question(self, body: dict[str, Any]) -> dict[str, Any]:
        topic = str(body.get("topic") or "RAG").strip() or "RAG"
        rounds = max(1, min(20, int(body.get("rounds") or 5)))
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        log = SessionLog(self.agent_config, session_id)
        round_index = len(_turn_events(log))
        if round_index >= rounds:
            return {"complete": True, "round": round_index, "rounds": rounds}
        state = prepare_interview_question(self.agent_config, topic, round_index)
        PENDING_QUESTIONS[session_id] = state
        evidence = (state.get("evidence_pack") or {}).get("evidence", [])
        return {
            "complete": False,
            "session_id": session_id,
            "round": round_index + 1,
            "rounds": rounds,
            "question": state["question"],
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
                "report": result["report"],
                "created_skills": result.get("created_skills", []),
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
            "report": result["report"],
            "created_skills": result.get("created_skills", []),
            "session_file": str(log.path),
        }

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
    input, textarea {
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
      </div>
      <div class="evidence" id="evidence"></div>
      <div class="output" id="report">Review will appear here.</div>
    </section>
  </main>
  <script>
    let sessionId = "";
    let topic = "RAG";
    let rounds = 5;
    const $ = id => document.getElementById(id);

    function setBusy(isBusy, text) {
      $("status").textContent = text || (isBusy ? "Working..." : "Ready");
      $("startBtn").disabled = isBusy;
      $("nextBtn").disabled = isBusy || !sessionId;
      $("submitBtn").disabled = isBusy || !sessionId;
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
        `<div class="score">Score: ${avg}/5</div>` +
        `<div>${escapeText("Missing points:\n" + (missing || "- none") + "\n\nBetter answer:\n" + (r.better_answer || "") + "\n\nNext tasks:\n" + (tasks || "- none"))}</div>`;
    }

    function escapeText(text) {
      return String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
    }

    async function startSession() {
      try {
        setBusy(true, "Creating session...");
        topic = $("topic").value.trim() || "RAG";
        rounds = Number($("rounds").value || 5);
        const data = await post("/api/session", {topic, rounds, session_id: $("session").value.trim()});
        sessionId = data.session_id;
        $("session").value = sessionId;
        $("sessionMeta").textContent = `${sessionId} · ${data.completed_rounds}/${data.rounds}`;
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
        const data = await post("/api/question", {session_id: sessionId, topic, rounds});
        if (data.complete) {
          $("question").textContent = "Session complete.";
          $("submitBtn").disabled = true;
          return;
        }
        $("question").textContent = `Round ${data.round}/${data.rounds}: ${data.question}`;
        $("answer").value = "";
        $("report").textContent = "Review will appear here.";
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
        $("sessionMeta").textContent = `${sessionId} · ${data.round}/${data.rounds}`;
        $("status").textContent = data.complete ? `Complete. Saved: ${data.session_file}` : "Scored. Click Next.";
      } catch (err) {
        $("status").innerHTML = `<span class="error">${escapeText(err.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    $("startBtn").addEventListener("click", startSession);
    $("nextBtn").addEventListener("click", nextQuestion);
    $("submitBtn").addEventListener("click", submitAnswer);
  </script>
</body>
</html>
"""
