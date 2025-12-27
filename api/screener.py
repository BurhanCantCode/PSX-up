"""
Stock Screener API Endpoint
Runs the exact backend screener inside E2B for parity with local.
"""

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

try:
    from e2b_code_interpreter import CodeInterpreter as Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
BACKEND_FILES = {
    "backend/__init__.py": "",
    "backend/stock_screener.py": (BACKEND_ROOT / "stock_screener.py").read_text(),
}
REQUIREMENTS_TEXT = (Path(__file__).parent.parent / "requirements.txt").read_text()

# Driver that calls backend.screen_stocks_api
SANDBOX_DRIVER = """
import asyncio
import json
import sys
from backend.stock_screener import screen_stocks_api, PSX_MAJOR_STOCKS

async def main(limit: int):
    symbols = PSX_MAJOR_STOCKS[:max(1, min(limit * 2, len(PSX_MAJOR_STOCKS)))]
    result = await screen_stocks_api(symbols)
    print(json.dumps(result))

if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(main(limit))
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            limit = 10
            if '?' in self.path:
                query = self.path.split('?')[1]
                for param in query.split('&'):
                    if param.startswith('limit='):
                        try:
                            limit = min(int(param.split('=')[1]), 40)
                        except Exception:
                            pass
            if not E2B_AVAILABLE:
                self._send_response(500, {'success': False, 'error': 'E2B SDK not available', 'top_picks': []})
                return
            result = self._run_screener(limit)
            self._send_response(200, result)
        except Exception as e:
            self._send_response(500, {'success': False, 'error': str(e), 'top_picks': []})

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _run_screener(self, limit: int) -> dict:
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
                execution = sbx.commands.run(
                    f"python /home/user/run.py {limit}",
                    timeout=420
                )
                if execution.exit_code == 0 and execution.stdout:
                    for line in execution.stdout.strip().split('\n'):
                        if line.strip().startswith('{'):
                            try:
                                return json.loads(line.strip())
                            except Exception:
                                continue
                return {
                    'success': False,
                    'error': execution.stderr or 'No output from screener',
                    'top_picks': []
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'top_picks': []
            }

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
