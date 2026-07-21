"""OpenAI-compatible proxy that puts GuardedTeacher between a client and a
raw model server (LM Studio, llama.cpp, vLLM — anything with a /v1 endpoint).

The model server itself has no concept of the leakage guard — it just serves
whatever weights are loaded (LM Studio's own chat window included). Point
clients at THIS proxy instead of the raw model server's port, and every
reply gets the same blocklist + judge check the leakage eval uses before it
goes out.

Binds to 127.0.0.1 by default — deliberately not reachable from other
devices unless HOST is overridden explicitly.

  BASE_MODEL=mythic-voice-9b-v3 BASE_URL=http://localhost:1234/v1 \
  BLOCKLIST=/path/to/blocklist.txt \
  JUDGE_MODEL=google/gemma-4-12B-it-qat-w4a16-ct JUDGE_URL=http://100.98.79.30:8203/v1 \
  PORT=8899 \
  python -m personaforge.serve_guarded
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from personaforge.guard import GuardedTeacher
from personaforge.teacher import Teacher


def build_guard() -> GuardedTeacher:
    base = Teacher(model=os.environ["BASE_MODEL"], base_url=os.environ["BASE_URL"])
    blocklist = [l.strip() for l in
                 Path(os.environ["BLOCKLIST"]).read_text().splitlines() if l.strip()]
    judge = None
    use_judge = False
    if os.environ.get("JUDGE_MODEL"):
        judge = Teacher(model=os.environ["JUDGE_MODEL"], base_url=os.environ["JUDGE_URL"])
        use_judge = True
    return GuardedTeacher(base, blocklist, judge=judge, use_judge=use_judge)


class Handler(BaseHTTPRequestHandler):
    guard: GuardedTeacher = None  # set on the class before serve_forever

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body["messages"]
        temperature = body.get("temperature", 0.8)
        max_tokens = body.get("max_tokens", 512)

        reply = self.guard.chat(messages, temperature=temperature, max_tokens=max_tokens)

        response = {
            "id": "guarded-0",
            "object": "chat.completion",
            "model": body.get("model", "guarded"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            payload = json.dumps({
                "object": "list",
                "data": [{"id": "guarded", "object": "model", "owned_by": "personaforge"}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # keep stdout clean; rely on the caller's own logging if needed


def main() -> None:
    Handler.guard = build_guard()
    port = int(os.environ.get("PORT", "8899"))
    host = os.environ.get("HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"guarded proxy listening on {host}:{port} -> {os.environ['BASE_URL']} "
          f"(judge={'on' if Handler.guard.use_judge else 'off'})")
    server.serve_forever()


if __name__ == "__main__":
    main()
