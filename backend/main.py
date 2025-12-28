#!/usr/bin/env python3
"""
PSX Fortune Teller API - FastAPI Backend
AI-powered stock predictions for Pakistan Stock Exchange
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

# ============================================================================
# SETUP
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import Stock Analyzer Logic
try:
    from backend.stock_analyzer_fixed import (
        check_data as check_stock_data,
        analyze_stock as start_stock_analysis,
        websocket_progress as stock_websocket,
        StockRequest,
        progress_data
    )
    ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Stock Analyzer Import Error: {e}")
    ANALYZER_AVAILABLE = False

# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="PSX Fortune Teller API",
    description="🔮 AI-powered stock predictions for Pakistan Stock Exchange",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Web Directory
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
    print(f"✅ Web directory mounted")

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Redirect to Fortune Teller UI"""
    return RedirectResponse(url="/analyzer")

@app.get("/analyzer")
async def analyzer_page():
    """Serve the Fortune Teller UI"""
    return FileResponse(WEB_DIR / "stock_analyzer.html")

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ============================================================================
# STOCK ANALYZER API
# ============================================================================

if ANALYZER_AVAILABLE:
    @app.post("/api/check-data")
    async def api_check_data(request: StockRequest):
        return await check_stock_data(request)

    @app.post("/api/analyze-stock")
    async def api_analyze_stock(request: StockRequest):
        return await start_stock_analysis(request)

    @app.websocket("/ws/progress/{job_id}")
    async def ws_progress(websocket: WebSocket, job_id: str):
        await stock_websocket(websocket, job_id)

    print("✅ Stock Analyzer routes registered")

# ============================================================================
# SCREENER & SENTIMENT
# ============================================================================

try:
    from backend.stock_screener import run_screener, screen_stock, PSX_MAJOR_STOCKS
    from backend.sentiment_analyzer import get_stock_sentiment
    from backend.hot_stocks import get_cached_trending_stocks, get_hot_stocks_for_homepage

    @app.get("/api/screener")
    async def screener(limit: int = 10):
        """Get top stock picks from screener"""
        try:
            results = run_screener(limit=min(limit, 50))
            return {"success": True, "top_picks": results}
        except Exception as e:
            return {"success": False, "error": str(e), "top_picks": []}

    @app.get("/api/trending")
    async def trending():
        """Get trending/hot stocks"""
        try:
            stocks = get_cached_trending_stocks()
            return {"success": True, "stocks": stocks}
        except Exception as e:
            return {"success": False, "error": str(e), "stocks": []}

    @app.get("/api/sentiment/{symbol}")
    async def sentiment(symbol: str):
        """Get AI sentiment analysis for a stock"""
        try:
            result = get_stock_sentiment(symbol.upper())
            return {"success": True, "sentiment": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    print("✅ Screener & Sentiment routes registered")
except ImportError as e:
    print(f"⚠️  Screener/Sentiment not available: {e}")

# ============================================================================
# HISTORY PERSISTENCE
# ============================================================================

@app.get("/api/history")
async def get_prediction_history():
    """Get list of saved Fortune Teller predictions (both SOTA and Research models)"""
    try:
        data_dir = BASE_DIR / "data"
        
        # Include both SOTA and Research model predictions
        sota_files = list(data_dir.glob("*_sota_predictions_2026.json"))
        research_files = list(data_dir.glob("*_research_predictions_2026.json"))
        all_files = sota_files + research_files
        
        # Sort by modification time (most recent first)
        all_files = sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)
        
        history = []
        seen_symbols = set()  # Only show most recent per symbol
        
        for f in all_files:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    symbol = data.get('symbol')
                    
                    # Skip if we already have a more recent analysis for this symbol
                    if symbol in seen_symbols:
                        continue
                    seen_symbols.add(symbol)
                    
                    daily_preds = data.get('daily_predictions', [])
                    last_pred = daily_preds[-1] if daily_preds else {}
                    first_pred = daily_preds[0] if daily_preds else {}
                    
                    history.append({
                        "filename": f.name,
                        "symbol": symbol,
                        "generated_at": data.get('generated_at'),
                        "current_price": first_pred.get('predicted_price', 0),
                        "predicted_return": last_pred.get('upside_potential', 0),
                    })
            except Exception:
                continue
        
        return {"success": True, "history": history}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/history/{filename}")
async def get_prediction_file(filename: str):
    """Get saved prediction with historical data for charting.
    
    Returns complete data structure matching WebSocket response format.
    """
    # Allow both SOTA and Research model prediction files
    valid_suffixes = ("_sota_predictions_2026.json", "_research_predictions_2026.json")
    if not filename.endswith(valid_suffixes) or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    file_path = BASE_DIR / "data" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Load historical data for charting
    symbol = data.get('symbol', '')
    hist_file = BASE_DIR / "data" / f"{symbol}_historical_with_indicators.json"
    historical_data = []
    
    if hist_file.exists():
        with open(hist_file, 'r') as hf:
            hist_raw = json.load(hf)
            historical_data = [{"Date": h['Date'], "Close": h['Close']} for h in hist_raw[-180:]]
    
    daily_preds = data.get('daily_predictions', [])
    first_pred = daily_preds[0] if daily_preds else {}
    
    # Get current price from historical data or first prediction
    current_price = 0
    if historical_data:
        current_price = historical_data[-1].get('Close', 0)
    if not current_price and first_pred:
        current_price = first_pred.get('predicted_price', 0)
    
    # Load cached sentiment if available
    sentiment = {}
    news_cache_file = BASE_DIR / "data" / "news_cache" / f"{symbol}_news.json"
    if news_cache_file.exists():
        try:
            with open(news_cache_file, 'r') as sf:
                sentiment_data = json.load(sf)
                sentiment = {
                    'signal': sentiment_data.get('signal', 'NEUTRAL'),
                    'signal_emoji': sentiment_data.get('signal_emoji', '🟡'),
                    'summary': sentiment_data.get('summary', 'No sentiment analysis available.'),
                    'sentiment_score': sentiment_data.get('sentiment_score', 0),
                    'confidence': sentiment_data.get('confidence', 0),
                    'recent_news': [
                        {
                            'title': n.get('title', ''),
                            'source': n.get('source_name', n.get('source', 'News')),
                            'published_at': n.get('date', 'Recently')
                        }
                        for n in sentiment_data.get('news_items', [])[:5]
                    ]
                }
        except Exception as e:
            print(f"Error loading sentiment cache: {e}")
    
    # Return complete structure matching WebSocket response
    return {
        "symbol": symbol,
        "current_price": current_price,
        "model": data.get('model', 'SOTA Ensemble'),
        "model_performance": data.get('metrics', {}),
        "daily_predictions": daily_preds,
        "historical_data": historical_data,
        "sentiment": sentiment,
        "prediction_reasoning": data.get('prediction_reasoning')  # Include reasoning if available
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("� PSX FORTUNE TELLER API")
    print("=" * 60)
    print("UI: http://localhost:8000/analyzer")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
