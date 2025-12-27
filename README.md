# 🔮 PSX Fortune Teller

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![ML](https://img.shields.io/badge/ML-Ensemble-orange?style=for-the-badge&logo=tensorflow)

**AI-Powered Stock Prediction System for Pakistan Stock Exchange**

<img width="1276" height="625" alt="image" src="https://github.com/user-attachments/assets/0d4b7c88-a15a-4804-9543-835a602f0900" />


*Daily predictions through December 2026 using state-of-the-art machine learning*

</div>

---

## 📊 Prediction Visualization

<div align="center">

```
                    Historical Data                    |        AI Predictions (2026)
                                                       |
    Price (PKR)                                        |
        ▲                                              |
   1000 │                                    ╭─────────┼──────────────────────────╮
        │                              ╭─────╯         |                          │
    900 │                        ╭─────╯               |     📈 Bullish Trend     │
        │                  ╭─────╯                     |                          │
    800 │            ╭─────╯                           |     Confidence: 85%      │
        │      ╭─────╯                                 |                          │
    700 │╭─────╯                                       |     Upside: +45.2%       │
        │                                              |                          │
    600 ├──────────────────────────────────────────────┼──────────────────────────┤
        │  Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec  │
        └──────────────────────────────────────────────┴──────────────────────────▶ Time
                                                       
                    ━━━ Historical    ╌╌╌ Forecast
```

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PSX Fortune Teller                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Web UI    │    │  FastAPI    │    │ SOTA Model  │    │  Sentiment  │  │
│  │  Chart.js   │◄──►│   Backend   │◄──►│  Ensemble   │◄──►│  Analyzer   │  │
│  │  WebSocket  │    │  WebSocket  │    │  6 Models   │    │  Groq LLM   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ML Pipeline Components                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  📈 Wavelet Denoising (db4 DWT)       | 📊 70+ Technical Indicators │   │
│  │  🔮 N-BEATS Trend Decomposition       | 🗓️ PSX Seasonal Features    │   │
│  │  ⚡ Exponential Gating (xLSTM-style)  | 🎯 Trend Dampening          │   │
│  │  📰 BR Research Article Scraping       | 💎 Quality Score System    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🧠 6-Model SOTA Ensemble
- **RandomForest** (500 estimators) - Robust baseline
- **ExtraTrees** (500 estimators) - Reduced variance
- **GradientBoosting** (500 estimators) - Sequential learning
- **XGBoost** - GPU-accelerated boosting
- **LightGBM** - Fast gradient boosting
- **Ridge Regression** - Regularized linear model

### 📊 Advanced Feature Engineering
- **Wavelet Denoising**: db4 DWT for noise reduction (50% → 70%+ accuracy)
- **N-BEATS Decomposition**: Polynomial trend + Fourier seasonality
- **PSX Seasonal Patterns**: Ramadan, EID, fiscal year effects
- **70+ Technical Indicators**: RSI, MACD, Bollinger Bands, etc.

### 🔮 Fortune Teller Enhancements
- **Deep Article Scraping**: Full Business Recorder research articles
- **Live Fundamentals**: P/E ratios, dividend yields from PSX Terminal
- **Quality Score System**: Identifies undervalued quality stocks
- **Trend Dampening**: Mean-reversion for quality stocks (prevents excessive bearishness)

### 🤖 AI Sentiment Analysis
- **Groq LLM** (Llama 3.3 70B) for news analysis
- Anti-hallucination guardrails
- Fundamental-aware predictions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/psx-prediction-app.git
cd psx-prediction-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (optional, for sentiment analysis)
```

### Running the Server

```bash
# Start the FastAPI server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the App

Open your browser and navigate to:
- **App**: http://localhost:8000/analyzer
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📁 Project Structure

```
psx-prediction-app/
├── backend/                      # FastAPI backend
│   ├── main.py                  # Main application & routes
│   ├── sota_model.py            # SOTA ensemble ML model
│   ├── stock_analyzer_fixed.py  # WebSocket analysis handler
│   ├── sentiment_analyzer.py    # AI sentiment analysis (Groq)
│   ├── article_scraper.py       # BR Research article scraper
│   └── sentiment_math.py        # Research-backed sentiment math
├── web/                          # Frontend
│   └── stock_analyzer.html      # Main UI with Chart.js
├── data/                         # Generated data (gitignored)
│   ├── *_historical_with_indicators.json
│   ├── *_sota_predictions_2026.json
│   └── news_cache/
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md
```

---

## 🔌 API Reference

### WebSocket: Stock Analysis

```
WS: /ws/progress/{job_id}
```

Real-time progress updates during analysis.

### POST /api/analyze-stock

Start a new stock analysis.

```bash
curl -X POST http://localhost:8000/api/analyze-stock \
  -H "Content-Type: application/json" \
  -d '{"symbol": "LUCK"}'
```

### GET /api/history

Get list of saved analyses.

```bash
curl http://localhost:8000/api/history
```

### GET /api/history/{filename}

Load a saved analysis with full data.

### GET /api/screener

Get top stock picks based on technical indicators.

### GET /api/sentiment/{symbol}

Get AI-powered sentiment analysis.

---

## 📈 Model Performance

| Metric | Typical Range | Description |
|--------|---------------|-------------|
| **R² Score** | 0.90 - 0.97 | Variance explained |
| **Trend Accuracy** | 65% - 75% | Direction prediction |
| **MASE** | < 1.0 | Better than naive forecast |
| **Sharpe Ratio** | 1.5 - 2.5 | Risk-adjusted returns |

---

## 🎯 Quality Score System

Stocks are scored based on fundamentals:

| Metric | Score Impact |
|--------|--------------|
| P/E < 8 | +0.15 (Deep Value) |
| P/E 8-12 | +0.10 (Value) |
| Dividend Yield > 8% | +0.15 (High Yield) |
| Dividend Yield 5-8% | +0.10 (Good Yield) |

**Quality Score > 0.55** triggers trend dampening to prevent excessive bearish predictions.

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for sentiment analysis | Optional |

---

## 📝 Disclaimer

> ⚠️ **This software is for educational and informational purposes only.**
> 
> It does not constitute financial advice. Stock market investments carry risk. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for Pakistan Stock Exchange traders**

🔮 *May your predictions be ever profitable* 🔮

</div>
