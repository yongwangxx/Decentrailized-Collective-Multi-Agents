"""Backend for any OpenAI-compatible chat-completions interface (standard library only).

Environment variables
    PA_BASE_URL     base address of the interface, e.g. https://<your-endpoint>/v1   (required)
    PA_API_KEY      the key                                                           (required)
    PA_MODEL        model identifier                                                  (required)
    PA_RPM          requests per minute sent by this process (default 100)
    PA_TEMPERATURE  requested sampling temperature (default 1.0)
    PA_MAX_TOKENS   limit of output tokens (default 20000)
    PA_EXTRA_BODY   JSON object merged into every request body, for fields that are specific to an interface or a
                    model (default: nothing)

A query is one conversation: an optional system message, the prompt as the user message and, if a tool is offered,
the tool in function-calling format with tool_choice "auto". When the model calls the tool, the program executes it,
appends the result, and asks again in the same conversation, at most `max_turns` requests. The key is read from the
environment at call time and is never written anywhere.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

from .base import new_reply, run_tool
from ..tools import openai_spec


class OpenAICompatBackend:
    name = "openai"

    def __init__(self):
        for v in ("PA_BASE_URL", "PA_API_KEY", "PA_MODEL"):
            if not os.environ.get(v):
                raise SystemExit(f"the openai backend needs the environment variable {v}")
        self.base = os.environ["PA_BASE_URL"].rstrip("/")
        self.model = os.environ["PA_MODEL"]
        self.rpm = float(os.environ.get("PA_RPM", 100))
        self.temperature = float(os.environ.get("PA_TEMPERATURE", 1.0))
        self.max_tokens = int(os.environ.get("PA_MAX_TOKENS", 20000))
        self.extra = json.loads(os.environ.get("PA_EXTRA_BODY", "{}"))
        self.stats = dict(sent=0, ok=0, rate_limited=0, errors=0)
        self._lock = threading.Lock()
        self._next = 0.0
        self._pause = 0.0

    def _slot(self):
        """Spread the requests of all threads evenly at PA_RPM; after a rate-limit response everybody pauses."""
        while True:
            with self._lock:
                now = time.time()
                t = max(self._next, self._pause, now)
                if t - now < 0.05:
                    self._next = t + 60.0 / self.rpm
                    self.stats["sent"] += 1
                    return
            time.sleep(min(t - now, 1.0))

    def _post(self, messages, tools):
        """One request; up to six tries on transport errors. Returns the decoded JSON or {"error": ...}."""
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "stream": False, **self.extra}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        last = "retries exhausted"
        for attempt in range(6):
            self._slot()
            try:
                req = urllib.request.Request(
                    self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                    headers={"Authorization": "Bearer " + os.environ["PA_API_KEY"], "Content-Type": "application/json",
                             "User-Agent": "particle-agents-repro/1.0"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    o = json.loads(r.read())
                with self._lock:
                    self.stats["ok"] += 1
                return o
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 429:
                    with self._lock:
                        self.stats["rate_limited"] += 1
                        self._pause = max(self._pause, time.time() + float(e.headers.get("Retry-After") or 30) + 2)
                else:
                    with self._lock:
                        self.stats["errors"] += 1
                    if e.code in (400, 401, 402, 403, 404):
                        break                                   # will not get better by repeating
                    time.sleep(3)
            except Exception as e:
                last = type(e).__name__
                with self._lock:
                    self.stats["errors"] += 1
                time.sleep(2)
        return {"error": last}

    def ask(self, prompt, tool=None, dice=None, system=None, max_turns=4):
        out = new_reply()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        tools = openai_spec(tool) if tool else None
        for _ in range(max_turns):
            o = self._post(messages, tools)
            out["n_requests"] += 1
            if "error" in o and "choices" not in o:
                out["error"] = str(o["error"])[:300]
                return out
            try:
                m = o["choices"][0]["message"]
            except Exception:
                out["error"] = "malformed response"
                return out
            out["model"] = o.get("model")
            out["reasoning"].append(m.get("reasoning_content") or m.get("reasoning") or "")
            tool_calls = m.get("tool_calls") or []
            if not tool_calls:
                out["text"] = (m.get("content") or "").strip()
                return out
            messages.append(m)
            for c in tool_calls:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = None
                reply = run_tool(tool, dice, fn.get("name"), args, out["calls"])
                messages.append({"role": "tool", "tool_call_id": c.get("id"), "content": reply})
        out["error"] = "max_turns"
        return out
