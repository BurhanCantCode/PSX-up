"""
Stock Screener API Endpoint
Quick market scan for top performing stocks.
"""

import os
import json
from http.server import BaseHTTPRequestHandler

try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

PSX_MAJOR_STOCKS = [
    'HBL', 'UBL', 'MCB', 'NBP', 'ABL', 'BAFL', 'BAHL', 'MEBL',
    'OGDC', 'PPL', 'POL', 'PSO', 'MARI',
    'LUCK', 'DGKC', 'MLCF', 'KOHC', 'FCCL', 'CHCC', 'PIOC',
    'ENGRO', 'FFC', 'EFERT', 'FATIMA',
    'HUBC', 'KEL', 'KAPCO',
    'SEARL', 'GLAXO',
    'INDU', 'PSMC', 'HCAR', 'MTL',
    'SYS', 'TRG', 'ISL', 'NESTLE'
]

SCREENER_SCRIPT = '''
import subprocess
import re
import json
import time
from datetime import datetime, timedelta
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "numpy", "--quiet"], capture_output=True)

import pandas as pd
import numpy as np

def fetch_month_data(symbol, month, year):
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol={symbol}"
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else None
    except:
        return None

def parse_html_table(html):
    rows = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)
    data = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
        if len(cells) >= 6:
            try:
                date_obj = datetime.strptime(cells[0].strip(), "%b %d, %Y")
                data.append({
                    'Date': date_obj.strftime('%Y-%m-%d'),
                    'Close': float(cells[4].strip().replace(',', '')),
                    'Volume': float(cells[5].strip().replace(',', ''))
                })
            except:
                continue
    return data

def screen_stock(symbol):
    data = []
    current_date = datetime.now()
    for i in range(3):
        target_date = current_date - timedelta(days=30 * i)
        html = fetch_month_data(symbol, target_date.month, target_date.year)
        if html:
            data.extend(parse_html_table(html))
        time.sleep(0.1)
    
    if not data or len(data) < 10:
        return None
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    current_price = df['Close'].iloc[-1]
    return_1w = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    return_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) >= 21 else return_1w
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 5 else 0
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 15 else 50
    
    sma_20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
    momentum_score = 2 if current_price > sma_20 else 0
    
    signal = 'BUY' if return_1m > 0 and momentum_score >= 2 else 'HOLD' if return_1m > -5 else 'SELL'
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'return_1w': round(return_1w, 2),
        'return_1m': round(return_1m, 2),
        'volatility': round(volatility, 2),
        'rsi': round(rsi, 1),
        'momentum_score': momentum_score,
        'signal': signal
    }

symbols = SYMBOLS_LIST
results = []
for symbol in symbols[:20]:
    result = screen_stock(symbol)
    if result:
        results.append(result)
    time.sleep(0.2)

results.sort(key=lambda x: x['return_1m'], reverse=True)
print(json.dumps({'success': True, 'top_picks': results[:10]}))
'''


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            limit = 10
            
            if '?' in self.path:
                query = self.path.split('?')[1]
                for param in query.split('&'):
                    if param.startswith('limit='):
                        try:
                            limit = min(int(param.split('=')[1]), 20)
                        except:
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
        """Run stock screener in E2B sandbox."""
        
        symbols_to_screen = PSX_MAJOR_STOCKS[:min(limit * 2, 40)]
        script = SCREENER_SCRIPT.replace('SYMBOLS_LIST', str(symbols_to_screen))
        
        try:
            with Sandbox() as sbx:
                sbx.files.write('/home/user/screener.py', script)
                execution = sbx.commands.run(
                    "python /home/user/screener.py",
                    timeout=180
                )
                
                if execution.exit_code == 0 and execution.stdout:
                    for line in execution.stdout.strip().split('\n'):
                        if line.strip().startswith('{'):
                            try:
                                return json.loads(line.strip())
                            except:
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
