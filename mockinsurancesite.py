from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import parse_qs
import hashlib

# Simple in-memory session store keyed by cookie
SESSIONS = {}

STYLE = """
<style>
  body { font-family: system-ui, sans-serif; max-width: 520px; margin: 40px auto; padding: 0 16px; color: #1f2937; }
  .card { border: 1px solid #d1d5db; border-radius: 8px; padding: 24px; background: #fff; }
  h2 { margin-top: 0; font-size: 1.25rem; }
  label { display: block; margin-top: 14px; font-weight: 600; font-size: 0.9rem; }
  input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; border: 1px solid #9ca3af; border-radius: 4px; }
  input[type="checkbox"] { width: auto; margin-right: 8px; }
  button { margin-top: 20px; padding: 10px 18px; background: #2563eb; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; }
  .error { background: #fee2e2; color: #991b1b; padding: 10px; border-radius: 4px; margin-bottom: 12px; font-size: 0.9rem; }
  .quote-box { background: #ecfdf5; border: 1px solid #10b981; padding: 16px; border-radius: 6px; }
  .hidden { display: none; }
</style>
"""


def render(title: str, body: str, error: str = "") -> bytes:
    err_html = f'<div role="alert" class="error">{error}</div>' if error else ""
    html = f"<!doctype html><html><head><title>{title}</title>{STYLE}</head><body><div class='card'><h2>{title}</h2>{err_html}{body}</div></body></html>"
    return html.encode("utf-8")


class InsuranceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # Keep terminal clean

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
            self.send_header("Set-Cookie", f"sid={sid}; Path=/")
        self.end_headers()
        self.wfile.write(content)
        self.wfile.flush()

    def _redirect(self, location: str, sid: str | None = None):
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        if sid:
            self.send_header("Set-Cookie", f"sid={sid}; Path=/")
        self.end_headers()
        self.wfile.flush()

    def do_GET(self):
        sid = self._get_sid()
        if self.path == "/" or self.path == "/step1":
            new_sid = hashlib.md5(str(id(self)).encode()).hexdigest()[:10]
            SESSIONS[new_sid] = {}
            body = """
            <form method="POST" action="/step1">
              <label for="full_name">Full Name</label>
              <input id="full_name" name="full_name" type="text" required />
              <label for="age">Age</label>
              <input id="age" name="age" type="number" required />
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 1 of 7: Personal Details", body), sid=new_sid)

        if self.path == "/step2":
            body = """
            <form method="POST" action="/step2">
              <label for="email">Email Address</label>
              <input id="email" name="email" type="email" required />
              <label for="city">City</label>
              <input id="city" name="city" type="text" required />
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 2 of 7: Contact Information", body))

        if self.path == "/step3":
            body = """
            <form method="POST" action="/step3">
              <label for="smoker">Do you smoke?</label>
              <select id="smoker" name="smoker" onchange="document.getElementById('cigs-wrap').classList.toggle('hidden', this.value !== 'yes')">
                <option value="no">no</option>
                <option value="yes">yes</option>
              </select>
              <div id="cigs-wrap" class="hidden">
                <label for="cigs_per_day">Cigarettes Per Day</label>
                <input id="cigs_per_day" name="cigs_per_day" type="number" />
              </div>
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 3 of 7: Health & Lifestyle", body))

        if self.path == "/step4":
            body = """
            <form method="POST" action="/step4">
              <label for="coverage_amount">Coverage Amount</label>
              <select id="coverage_amount" name="coverage_amount">
                <option value="500000">500000</option>
                <option value="750000">750000</option>
                <option value="1000000">1000000</option>
              </select>
              <label for="plan_type">Plan Tier</label>
              <select id="plan_type" name="plan_type">
                <option value="basic">basic</option>
                <option value="plus">plus</option>
              </select>
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 4 of 7: Coverage Selection", body))

        if self.path == "/step5":
            body = """
            <form method="POST" action="/step5">
              <label for="deductible">Annual Deductible</label>
              <select id="deductible" name="deductible">
                <option value="500">500</option>
                <option value="1000">1000</option>
                <option value="2000">2000</option>
              </select>
              <label for="term_years">Policy Term (Years)</label>
              <input id="term_years" name="term_years" type="number" value="10" required />
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 5 of 7: Deductible & Term", body))

        if self.path == "/step6":
            body = """
            <form method="POST" action="/step6">
              <label for="nominee_name">Nominee Full Name</label>
              <input id="nominee_name" name="nominee_name" type="text" required />
              <label for="nominee_relation">Relationship</label>
              <select id="nominee_relation" name="nominee_relation">
                <option value="spouse">spouse</option>
                <option value="parent">parent</option>
                <option value="child">child</option>
              </select>
              <button id="btn-next" type="submit">Next Step</button>
            </form>"""
            return self._send(render("Step 6 of 7: Nominee Details", body))

        if self.path == "/step7":
            body = """
            <form method="POST" action="/step7">
              <label><input id="agree_terms" name="agree_terms" type="checkbox" value="yes" /> I confirm all provided details are accurate.</label>
              <button id="btn-submit" type="submit">Calculate Premium</button>
            </form>"""
            return self._send(render("Step 7 of 7: Final Declaration", body))

        if self.path == "/quote":
            data = SESSIONS.get(sid, {})
            age = int(data.get("age", 30))
            cov = int(data.get("coverage_amount", 500000))
            smoker = data.get("smoker", "no")
            plan = data.get("plan_type", "basic")
            deductible = int(data.get("deductible", 500))

            premium = round(
                (cov / 1000) * 0.5
                + (age * 4)
                + (300 if smoker == "yes" else 0)
                + (100 if plan == "plus" else 0)
                - (deductible * 0.05),
                2,
            )
            qid = "Q-" + hashlib.md5(f"{sid}{premium}".encode()).hexdigest()[:6].upper()
            body = f"""
            <div class="quote-box">
              <p><strong>Applicant:</strong> {data.get('full_name', 'N/A')}</p>
              <p><strong>Quote Reference:</strong> {qid}</p>
              <p><strong>Annual Premium:</strong> ${premium:.2f}</p>
            </div>"""
            return self._send(render("Your Insurance Quote Summary", body))

        self.send_error(404)

    def do_POST(self):
        sid = self._get_sid()
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(raw).items()}
        SESSIONS.setdefault(sid, {}).update(form)

        if self.path == "/step1":
            if not form.get("full_name") or not form.get("age"):
                return self._send(render("Step 1 of 7: Personal Details", "<a href='/step1'>Retry</a>", "Name and Age are required."))
            return self._redirect("/step2")
        if self.path == "/step2":
            return self._redirect("/step3")
        if self.path == "/step3":
            if form.get("smoker") == "yes" and not form.get("cigs_per_day"):
                body = """
                <form method="POST" action="/step3">
                  <label for="smoker">Do you smoke?</label>
                  <select id="smoker" name="smoker"><option value="yes" selected>yes</option><option value="no">no</option></select>
                  <label for="cigs_per_day">Cigarettes Per Day</label>
                  <input id="cigs_per_day" name="cigs_per_day" type="number" />
                  <button id="btn-next" type="submit">Next Step</button>
                </form>"""
                return self._send(render("Step 3 of 7: Health & Lifestyle", body, "Please enter cigarettes per day for smokers."))
            return self._redirect("/step4")
        if self.path == "/step4":
            return self._redirect("/step5")
        if self.path == "/step5":
            return self._redirect("/step6")
        if self.path == "/step6":
            return self._redirect("/step7")
        if self.path == "/step7":
            if form.get("agree_terms") != "yes":
                body = """
                <form method="POST" action="/step7">
                  <label><input id="agree_terms" name="agree_terms" type="checkbox" value="yes" /> I confirm all provided details are accurate.</label>
                  <button id="btn-submit" type="submit">Calculate Premium</button>
                </form>"""
                return self._send(render("Step 7 of 7: Final Declaration", body, "You must accept the terms to calculate a quote."))
            return self._redirect("/quote")


def start_server_in_thread(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), InsuranceHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def main():
    print("Mock Insurance Portal running at http://127.0.0.1:8000/step1")
    httpd = ThreadingHTTPServer(("127.0.0.1", 8000), InsuranceHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
