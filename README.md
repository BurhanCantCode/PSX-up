# 🔮 PSX-up

AI-powered stock prediction system for Pakistan Stock Exchange (PSX) using ensemble machine learning with N-BEATS decomposition, wavelet denoising, and multi-horizon forecasting.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔮 Fortune Teller Engine**: Causal O(n) ensemble model with XGBoost, LightGBM, and CatBoost
- **📈 Daily Predictions**: Forecasts through December 2026 with confidence intervals
- **📰 AI Sentiment**: Groq-powered news sentiment analysis with mathematical adjustments
- **🎨 Modern UI**: Clean "Investo" design with interactive charts
- **⚡ Fast**: Optimized sliding window operations (1-2s vs 30-60s)

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/BurhanCantCode/PSX-up.git
cd PSX-up

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure (for sentiment analysis)
cp .env.example .env
# Add your GROQ_API_KEY

# Run
python backend/main.py
```

Open **http://localhost:8000/analyzer**

## 📁 Project Structure

```
psx-fortune-teller/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── sota_model.py           # 🔮 Fortune Teller ML engine
│   ├── stock_analyzer_fixed.py # WebSocket analysis flow
│   ├── sentiment_analyzer.py   # Groq AI sentiment
│   ├── sentiment_math.py       # Adjustment calculations
│   ├── stock_screener.py       # Market scanner
│   └── hot_stocks.py           # Trending stocks
├── web/
│   └── stock_analyzer.html     # Frontend UI
├── data/                       # Generated predictions
├── requirements.txt
└── .env
```

## 🔬 Model Architecture

The Fortune Teller uses a **causal multi-horizon ensemble**:

1. **N-BEATS Decomposition**: Trend + weekly/monthly seasonality
2. **Wavelet Denoising**: db4 wavelet with sliding window
3. **Macro Features**: Holiday effects, day-of-week, month patterns
4. **Ensemble**: XGBoost (40%) + LightGBM (35%) + CatBoost (25%)
5. **Horizon Weighting**: Dynamic weights for short/medium/long-term

All operations are **strictly causal** (no look-ahead bias).

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyzer` | GET | Fortune Teller UI |
| `/api/analyze-stock` | POST | Run full analysis |
| `/api/history` | GET | Saved predictions |
| `/api/screener` | GET | Top stock picks |
| `/api/trending` | GET | Hot stocks |
| `/api/sentiment/{symbol}` | GET | AI sentiment |

## ⚙️ Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key  # Required for sentiment analysis
```

## 📈 Metrics

- **Trend Accuracy**: ~86%
- **R² Score**: >0.99
- **MASE**: <0.3

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ for PSX traders
