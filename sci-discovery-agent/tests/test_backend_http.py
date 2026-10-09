"""End-to-end tests of OpenAICompatBackend against a local fake server.

The fake server is a *test double* that returns canned replies. These tests verify the
HTTP plumbing and the agent loop over a real socket; they say nothing about LLM quality.
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from scidisc.agent import DiscoveryAgent
from scidisc.llm import BackendError, OpenAICompatBackend
from scidisc.tasks import get_task


def make_server(replies, status=200):
    log = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
            log["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            if status != 200:
                self.send_response(status)
                self.end_headers()
                return
            text = replies[min(len(log["requests"]) - 1, len(replies) - 1)]
            payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log


class HttpBackendTests(unittest.TestCase):
    def test_full_agent_run_over_http(self):
        reply = json.dumps({"candidates": [{"expr": "c0*a**c1", "why": "power law"}]})
        srv, log = make_server([reply])
        self.addCleanup(srv.shutdown)
        be = OpenAICompatBackend(model="fake-local", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        task = get_task("kepler")
        res = DiscoveryAgent(be, max_iters=2).solve(task, task.make_dataset(0), 0)

        self.assertGreater(res.val_r2, 0.999)
        req = log["requests"][0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["body"]["model"], "fake-local")
        self.assertEqual(req["body"]["messages"][0]["role"], "system")
        self.assertEqual(req["body"]["messages"][1]["role"], "user")
        self.assertTrue(req["auth"].startswith("Bearer "))
        self.assertEqual(be.calls, 1)
        self.assertGreater(be.chars_in, 0)

    def test_http_error_becomes_backend_error(self):
        srv, _ = make_server(["x"], status=500)
        self.addCleanup(srv.shutdown)
        be = OpenAICompatBackend(model="m", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        with self.assertRaises(BackendError):
            be.complete("sys", [{"role": "user", "content": "hi"}])

    def test_connection_refused_becomes_backend_error(self):
        be = OpenAICompatBackend(model="m", base_url="http://127.0.0.1:9/v1", timeout=2)
        with self.assertRaises(BackendError):
            be.complete("sys", [{"role": "user", "content": "hi"}])

    def test_agent_survives_dead_server(self):
        be = OpenAICompatBackend(model="m", base_url="http://127.0.0.1:9/v1", timeout=2)
        task = get_task("kepler")
        res = DiscoveryAgent(be, max_iters=3).solve(task, task.make_dataset(0), 0)
        self.assertEqual(res.formula, "<none>")
        self.assertTrue(res.trace and "error" in res.trace[0])


if __name__ == "__main__":
    unittest.main()
