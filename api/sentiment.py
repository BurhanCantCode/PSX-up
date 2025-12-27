"""
Sentiment Analysis API Endpoint
Uses Groq LLM for news sentiment analysis.
This runs directly on Vercel (not in E2B) since it's lightweight.
"""

import os
import json
import subprocess
import re
from http.server import BaseHTTPRequestHandler
from datetime import datetime
from typing import List, Dict

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Stock symbol to company name mapping
STOCK_COMPANIES = {
    'LUCK': ('Lucky Cement', 'Lucky Cement Limited'),
    'HBL': ('Habib Bank', 'Habib Bank Limited'),
    'UBL': ('United Bank', 'United Bank Limited'),
    'MCB': ('MCB Bank', 'MCB Bank Limited'),
    'OGDC': ('OGDC', 'Oil and Gas Development Company'),
    'PPL': ('Pakistan Petroleum', 'Pakistan Petroleum Limited'),
    'PSO': ('Pakistan State Oil', 'PSO'),
    'ENGRO': ('Engro', 'Engro Corporation'),
    'FFC': ('Fauji Fertilizer', 'FFC'),
    'FATIMA': ('Fatima Fertilizer', 'Fatima Group'),
    'HUBC': ('Hub Power', 'HUBCO'),
    'SYS': ('Systems Limited', 'Systems Ltd'),
    'TRG': ('TRG Pakistan', 'TRG'),
    'NESTLE': ('Nestle Pakistan', 'Nestle'),
    'MARI': ('Mari Petroleum', 'Mari Gas'),
    'KEL': ('K-Electric', 'Karachi Electric'),
    'GLAXO': ('GlaxoSmithKline', 'GSK Pakistan'),
}


def fetch_psx_announcements(symbol: str) -> List[Dict]:
    """Fetch PSX company announcements."""
    news_items = []
    
    try:
        url = f"https://dps.psx.com.pk/company/{symbol.upper()}"
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '15', url],
            capture_output=True, text=True, timeout=20
        )
        
        if result.returncode == 0 and result.stdout:
            html = result.stdout
            rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL | re.IGNORECASE)
            
            for row in rows[:10]:
                cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
                if len(cells) >= 2:
                    date_text = cells[0].strip()
                    title = ' '.join(cells[1:3]).strip()
                    
                    if len(title) > 10:
                        news_items.append({
                            'title': title[:200],
                            'date': date_text,
                            'source': 'PSX',
                            'url': url
                        })
    except Exception:
        pass
    
    return news_items[:5]


def analyze_with_groq(symbol: str, company_name: str, news_items: List[Dict]) -> Dict:
    """Use Groq LLM for sentiment analysis."""
    
    if not GROQ_AVAILABLE:
        return fallback_analysis(news_items)
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return fallback_analysis(news_items)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if news_items:
        news_text = "\n".join([
            f"- [{item.get('date', 'unknown')}] [{item.get('source', 'Unknown')}] {item['title']}"
            for item in news_items[:10]
        ])
    else:
        news_text = "No recent news found."
    
    prompt = f"""You are a conservative Pakistani stock market analyst. Today is {current_date}.

RULES:
1. Only cite facts from the headlines below
2. Never fabricate deals or acquisitions not mentioned
3. Never give specific percentage predictions
4. Be conservative and acknowledge uncertainty

Analyze news about {symbol} ({company_name}):

{news_text}

Respond in JSON:
{{
    "sentiment_score": <-1.0 to +1.0>,
    "signal": "<BULLISH|NEUTRAL|BEARISH>",
    "confidence": <0.0-1.0>,
    "key_events": ["events from headlines"],
    "risks": ["risks mentioned"],
    "catalysts": ["catalysts from headlines"],
    "summary": "<2-3 factual sentences>"
}}

Return ONLY valid JSON."""

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.1
        )
        
        result = json.loads(completion.choices[0].message.content.strip())
        result['model'] = 'llama-3.3-70b-versatile'
        result['analyzed_at'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        return fallback_analysis(news_items)


def fallback_analysis(news_items: List[Dict]) -> Dict:
    """Keyword-based fallback when Groq unavailable."""
    text = ' '.join([item.get('title', '') for item in news_items]).lower()
    
    bullish = sum(1 for w in ['profit', 'growth', 'dividend', 'acquire', 'expansion', 'record', 'surge'] if w in text)
    bearish = sum(1 for w in ['loss', 'decline', 'drop', 'fraud', 'investigation', 'shutdown'] if w in text)
    
    if bullish > bearish:
        score, signal = min(0.5, bullish * 0.15), 'BULLISH'
    elif bearish > bullish:
        score, signal = max(-0.5, -bearish * 0.15), 'BEARISH'
    else:
        score, signal = 0, 'NEUTRAL'
    
    return {
        'sentiment_score': score,
        'signal': signal,
        'confidence': 0.3,
        'key_events': [],
        'summary': 'Analysis based on keyword matching (LLM unavailable)',
        'model': 'fallback',
        'analyzed_at': datetime.now().isoformat()
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Extract symbol from path: /api/sentiment/LUCK
            path_parts = self.path.split('/')
            symbol = None
            
            for i, part in enumerate(path_parts):
                if part == 'sentiment' and i + 1 < len(path_parts):
                    symbol = path_parts[i + 1].split('?')[0].upper()
                    break
            
            if not symbol:
                self._send_response(400, {'success': False, 'error': 'Symbol required'})
                return
            
            company_names = STOCK_COMPANIES.get(symbol, (symbol,))
            company_name = company_names[0]
            
            # Fetch news
            news_items = fetch_psx_announcements(symbol)
            
            # Analyze with Groq
            analysis = analyze_with_groq(symbol, company_name, news_items)
            
            # Build response
            signal = analysis.get('signal', 'NEUTRAL')
            if signal == 'BULLISH':
                signal_emoji = '[+]'
            elif signal == 'BEARISH':
                signal_emoji = '[-]'
            else:
                signal_emoji = '[=]'
            
            result = {
                'success': True,
                'sentiment': {
                    'symbol': symbol,
                    'company': company_name,
                    'signal': signal,
                    'signal_emoji': signal_emoji,
                    'sentiment_score': analysis.get('sentiment_score', 0),
                    'confidence': analysis.get('confidence', 0),
                    'summary': analysis.get('summary', ''),
                    'key_events': analysis.get('key_events', []),
                    'risks': analysis.get('risks', []),
                    'catalysts': analysis.get('catalysts', []),
                    'news_count': len(news_items),
                    'recent_news': news_items,
                    'model': analysis.get('model', 'unknown'),
                    'analyzed_at': analysis.get('analyzed_at', '')
                }
            }
            
            self._send_response(200, result)
            
        except Exception as e:
            self._send_response(500, {'success': False, 'error': str(e)})
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
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

