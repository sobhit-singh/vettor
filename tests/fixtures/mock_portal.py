from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

# In-memory session store keyed by cookie / sid
SESSIONS: dict[str, dict[str, str]] = {}

STYLE = """
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 540px; margin: 40px auto; padding: 0 16px; color: #1f2937; background: #f9fafb; }
  .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 24px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  h2 { margin-top: 0; font-size: 1.25rem; color: #111827; }
  label { display: block; margin-top: 14px; font-weight: 600; font-size: 0.9rem; color: #374151; }
  input, select, textarea { width: 100%; padding: 8px 12px; margin-top: 4px; box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.95rem; }
  input[type="checkbox"] { width: auto; margin-right: 8px; display: inline-block; vertical-align: middle; }
  .checkbox-label { display: flex; align-items: center; margin-top: 16px; font-weight: normal; cursor: pointer; }
  button { margin-top: 20px; padding: 10px 20px; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.95rem; width: 100%; }
  button:hover { background: #1d4ed8; }
  .error { background: #fee2e2; color: #991b1b; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 0.9rem; border: 1px solid #fecaca; }
  .quote-box { background: #ecfdf5; border: 1px solid #a7f3d0; padding: 20px; border-radius: 6px; margin-top: 16px; }
  .quote-box p { margin: 8px 0; }
  .price { font-size: 1.5rem; font-weight: 700; color: #047857; }
  .hidden { display: none; }
  .review-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  .review-table td { padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
  .review-table td:first-child { font-weight: 600; color: #4b5563; width: 40%; }
</style>
"""


def render(title: str, body: str, error: str = "") -> bytes:
    err_html = f'<div role="alert" class="error">{error}</div>' if error else ""
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  {STYLE}
</head>
<body>
  <div class="card">
    <h2>{title}</h2>
    {err_html}
    {body}
  </div>
</body>
</html>"""
    return html.encode("utf-8")


class MockPortalHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # Suppress request spam

    def _get_sid(self) -> str:
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            if "sid=" in part:
                return part.split("sid=")[1].strip()
        return "default"

    def _send(self, content: bytes, sid: str | None = None):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Connection", "close")
        if sid:
            self.send_header("Set-Cookie", f"sid={sid}; Path=/; HttpOnly")
        self.end_headers()
        self.wfile.write(content)
        self.wfile.flush()

    def _redirect(self, location: str, sid: str | None = None):
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        if sid:
            self.send_header("Set-Cookie", f"sid={sid}; Path=/; HttpOnly")
        self.end_headers()
        self.wfile.flush()

    def do_GET(self):
        sid = self._get_sid()

        if self.path in ("/", "/step1"):
            new_sid = hashlib.md5(f"{id(self)}-{self.client_address}".encode()).hexdigest()[:12]
            SESSIONS[new_sid] = {}
            body = """
            <form method="POST" action="/step1">
              <label for="name">Full Name</label>
              <input id="name" name="name" type="text" placeholder="e.g. Alex Morgan" required />

              <label for="age">Age</label>
              <input id="age" name="age" type="number" placeholder="e.g. 28" required
                     oninput="document.getElementById('young-adult-box').classList.toggle('hidden', parseInt(this.value, 10) >= 25 || isNaN(parseInt(this.value, 10)))" />

              <div id="young-adult-box" class="hidden">
                <label for="student_id">Student ID (Under 25 discount)</label>
                <input id="student_id" name="student_id" type="text" placeholder="STU-12345" />
              </div>

              <button id="btn-next" type="submit">Continue to Step 2</button>
            </form>
            """
            return self._send(render("Step 1: Personal Details", body), sid=new_sid)

        if self.path == "/step2":
            body = """
            <form method="POST" action="/step2">
              <label for="plan">Plan Tier</label>
              <select id="plan" name="plan">
                <option value="Basic">Basic ($150/yr)</option>
                <option value="Pro" selected>Pro ($250/yr)</option>
                <option value="Enterprise">Enterprise ($500/yr)</option>
              </select>

              <label for="notes">Additional Notes</label>
              <textarea id="notes" name="notes" rows="3" placeholder="Optional comments or requirements..."></textarea>

              <label class="checkbox-label" for="agree_terms">
                <input id="agree_terms" name="agree_terms" type="checkbox" value="yes" required />
                I agree to the terms and privacy conditions
              </label>

              <button id="btn-next" type="submit">Continue to Review</button>
            </form>
            """
            return self._send(render("Step 2: Plan Selection & Terms", body))

        if self.path == "/step3":
            data = SESSIONS.get(sid, {})
            name = data.get("name", "N/A")
            age = data.get("age", "N/A")
            plan = data.get("plan", "Basic")
            notes = data.get("notes", "None")
            body = f"""
            <table class="review-table">
              <tr><td>Full Name</td><td>{name}</td></tr>
              <tr><td>Age</td><td>{age}</td></tr>
              <tr><td>Selected Plan</td><td>{plan}</td></tr>
              <tr><td>Notes</td><td>{notes}</td></tr>
            </table>

            <form method="POST" action="/step3">
              <button id="btn-calculate" type="submit">Calculate Quote</button>
            </form>
            """
            return self._send(render("Step 3: Review Details", body))

        if self.path == "/quote":
            data = SESSIONS.get(sid, {})
            name = data.get("name", "Alex Morgan")
            plan = data.get("plan", "Pro")

            price = 250.00
            if plan == "Basic":
                price = 150.00
            elif plan == "Enterprise":
                price = 500.00

            try:
                if int(data.get("age", "30")) < 25 and data.get("student_id"):
                    price *= 0.85
            except ValueError:
                pass

            ref_code = "Q-" + hashlib.md5(f"{sid}-{price}".encode()).hexdigest()[:6].upper()
            body = f"""
            <div class="quote-box" id="quote-summary">
              <p><strong>Status:</strong> Quote Calculated Successfully</p>
              <p><strong>Applicant:</strong> {name}</p>
              <p><strong>Tier:</strong> {plan}</p>
              <p><strong>Reference Number:</strong> <span id="ref-code">{ref_code}</span></p>
              <p><strong>Calculated Price:</strong> <span class="price">${price:.2f}</span></p>
            </div>
            <p style="text-align: center; margin-top: 20px; color: #6b7280; font-size: 0.9rem;">
              Confirmation email sent. Thank you for using Vettor Mock Portal.
            </p>
            """
            return self._send(render("Quote Summary", body))

        self.send_error(404, "Page not found")

    def do_POST(self):
        sid = self._get_sid()
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(raw).items()}
        SESSIONS.setdefault(sid, {}).update(form)

        if self.path == "/step1":
            if not form.get("name") or not form.get("age"):
                return self._send(render("Step 1: Personal Details", "<p><a href='/step1'>Retry</a></p>", "Name and Age are required."))
            return self._redirect("/step2")

        if self.path == "/step2":
            if form.get("agree_terms") != "yes":
                return self._send(render("Step 2: Plan Selection & Terms", "<p><a href='/step2'>Retry</a></p>", "You must agree to terms."))
            return self._redirect("/step3")

        if self.path == "/step3":
            return self._redirect("/quote")

        self.send_error(404, "Unknown action")


def start_server_in_thread(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Starts the mock portal in a background daemon thread and returns (server, base_url)."""
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), MockPortalHandler)
    actual_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{actual_port}"
    return httpd, base_url


def main():
    httpd, url = start_server_in_thread(port=8000)
    print(f"Mock Portal running at {url}/step1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
