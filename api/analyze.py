"""
Stock Analysis API Endpoint
Triggers E2B sandbox for ML computations and returns results.
Ships the exact backend scripts into the sandbox (no compromises).
"""

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

try:
    # CodeInterpreter replaces Sandbox; matches current E2B SDK
    from e2b_code_interpreter import CodeInterpreter as Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

# Read backend sources verbatim so the sandbox uses the exact same code
BACKEND_ROOT = Path(__file__).parent.parent / "backend"
BACKEND_FILES = {
    "backend/__init__.py": "",
    "backend/stock_analyzer_fixed.py": (BACKEND_ROOT / "stock_analyzer_fixed.py").read_text(),
    "backend/sota_model.py": (BACKEND_ROOT / "sota_model.py").read_text(),
}
REQUIREMENTS_TEXT = (Path(__file__).parent.parent / "requirements.txt").read_text()

# Driver script that uses the backend code without modification
SANDBOX_DRIVER = """
import asyncio
import json
import sys
import pandas as pd

from backend.stock_analyzer_fixed import fetch_historical_data_async, load_data
from backend.sota_model import SOTAEnsemblePredictor, PYWT_AVAILABLE


async def main(symbol: str):
    symbol = symbol.upper()
    # Fetch data (identical logic to backend)
    await fetch_historical_data_async(symbol)
    df = load_data(symbol)
    if df is None or df.empty:
        print(json.dumps({'status': 'error', 'symbol': symbol, 'error': f'No data found for {symbol}'}))
        return

    # Train + predict with full SOTA model
    model = SOTAEnsemblePredictor(lookback=150, horizon=21, use_wavelet=PYWT_AVAILABLE)
    metrics = model.fit(df, verbose=False)
    daily_predictions = model.predict_daily(df, end_date='2026-12-31')

    history_df = df.tail(180)[['Date', 'Close']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date']).dt.strftime('%Y-%m-%d')

    print(json.dumps({
        'status': 'complete',
        'symbol': symbol,
        'current_price': float(df['Close'].iloc[-1]),
        'data_points': len(df),
        'model': 'SOTA Ensemble (RF, ET, GB, XGBoost, LightGBM, Ridge) - 500 estimators',
        'model_performance': {k: float(v) for k, v in metrics.items()},
        'wavelet_denoising': PYWT_AVAILABLE,
        'features_used': len(model.feature_names),
        'daily_predictions': daily_predictions,
        'historical_data': history_df.to_dict('records'),
        'generated_at': pd.Timestamp.utcnow().isoformat()
    }))


if __name__ == '__main__':
    sym = sys.argv[1] if len(sys.argv) > 1 else 'LUCK'
    asyncio.run(main(sym))
"""


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            symbol = body.get('symbol', '').upper().strip()
            
            if not symbol:
                self._send_response(400, {'error': 'Symbol is required'})
                return
            
            if not symbol.isalpha() or len(symbol) > 10:
                self._send_response(400, {'error': 'Invalid symbol format'})
                return
            
            if not E2B_AVAILABLE:
                self._send_response(500, {'error': 'E2B SDK not available'})
                return
            
            result = self._run_analysis(symbol)
            self._send_response(200, result)
            
        except json.JSONDecodeError:
            self._send_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_response(500, {'error': str(e)})
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def _run_analysis(self, symbol: str) -> dict:
        """Run stock analysis in E2B sandbox."""
        
        try:
            with Sandbox() as sbx:
                # Write backend sources verbatim
                for path, content in BACKEND_FILES.items():
                    sbx.files.write(f"/home/user/{path}", content)

                # Write requirements and install everything (exactly as project)
                sbx.files.write("/home/user/requirements.txt", REQUIREMENTS_TEXT)
                sbx.commands.run(
                    "pip install -r /home/user/requirements.txt --quiet",
                    timeout=180
                )

                # Write driver that calls the backend code directly
                sbx.files.write("/home/user/run.py", SANDBOX_DRIVER)

                execution = sbx.commands.run(
                    f"python /home/user/run.py {symbol}",
                    timeout=420
                )
                
                if execution.exit_code == 0 and execution.stdout:
                    for line in execution.stdout.strip().split('\n'):
                        if line.strip().startswith('{'):
                            try:
                                return json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue
                    return {
                        'status': 'error',
                        'error': 'No valid JSON output',
                        'raw': execution.stdout[:500]
                    }
                else:
                    return {
                        'status': 'error',
                        'error': execution.stderr or 'Analysis failed',
                        'exit_code': execution.exit_code
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'error': f'E2B execution error: {str(e)}'
            }
    
    def _send_response(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
