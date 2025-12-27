"""
Sentiment Analysis API Endpoint
Runs the exact backend sentiment code inside E2B for parity with local.
"""

import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler

try:
    from e2b_code_interpreter import CodeInterpreter as Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

# Ship backend sources verbatim
BACKEND_ROOT = Path(__file__).parent.parent / "backend"
BACKEND_FILES = {
    "backend/__init__.py": "",
    "backend/sentiment_analyzer.py": (BACKEND_ROOT / "sentiment_analyzer.py").read_text(),
    "backend/sentiment_math.py": (BACKEND_ROOT / "sentiment_math.py").read_text(),
}
REQUIREMENTS_TEXT = (Path(__file__).parent.parent / "requirements.txt").read_text()

# Driver that calls the backend sentiment analyzer directly
SANDBOX_DRIVER = """
import json
import sys
from backend.sentiment_analyzer import get_stock_sentiment

def main(symbol: str):
    result = get_stock_sentiment(symbol, use_cache=False)
    print(json.dumps({'success': True, 'sentiment': result}))

if __name__ == '__main__':
    sym = sys.argv[1] if len(sys.argv) > 1 else 'LUCK'
    main(sym)
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            path_parts = self.path.split('/')
            symbol = None
            for i, part in enumerate(path_parts):
                if part == 'sentiment' and i + 1 < len(path_parts):
                    symbol = path_parts[i + 1].split('?')[0].upper()
                    break
            if not symbol:
                self._send_response(400, {'success': False, 'error': 'Symbol required'})
                return
            if not E2B_AVAILABLE:
                self._send_response(500, {'success': False, 'error': 'E2B SDK not available'})
                return
            result = self._run_sentiment(symbol)
            self._send_response(200, result)
        except Exception as e:
            self._send_response(500, {'success': False, 'error': str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _run_sentiment(self, symbol: str) -> dict:
        try:
            with Sandbox() as sbx:
                for path, content in BACKEND_FILES.items():
                    sbx.files.write(f"/home/user/{path}", content)
                sbx.files.write("/home/user/requirements.txt", REQUIREMENTS_TEXT)
                sbx.commands.run(
                    "pip install -r /home/user/requirements.txt --quiet",
                    timeout=180
                )
                sbx.files.write("/home/user/run.py", SANDBOX_DRIVER)
                env = {}
                if os.getenv("GROQ_API_KEY"):
                    env["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
                execution = sbx.commands.run(
                    f"python /home/user/run.py {symbol}",
                    timeout=180,
                    env=env or None
                )
                if execution.exit_code == 0 and execution.stdout:
                    for line in execution.stdout.strip().split('\n'):
                        if line.strip().startswith('{'):
                            try:
                                return json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue
                return {'success': False, 'error': execution.stderr or 'No output'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _send_response(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
