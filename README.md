# PSX Stock Predictor

A production-grade stock prediction system for Pakistan Stock Exchange (PSX) using state-of-the-art machine learning ensemble models. Deployed on Vercel with E2B sandboxed execution for ML computations.

## Architecture

```
Frontend (Vercel Static)  -->  API Layer (Vercel Serverless)  -->  E2B Sandbox (ML Execution)
         |                              |                                   |
    index.html                   Python Functions                 Full ML Pipeline
    Chart.js                     E2B SDK Integration              scikit-learn
                                 CORS Handling                    XGBoost, LightGBM
                                                                  PyWavelets
```

## Tech Stack

### Frontend
- Vanilla JavaScript with Chart.js for visualization
- CSS Grid/Flexbox layout
- Responsive design

### API Layer (Vercel Serverless)
- Python 3.9+ runtime
- E2B Code Interpreter SDK
- Groq SDK for sentiment analysis

### ML Pipeline (E2B Sandbox)
- **Data Source**: PSX Historical Data API
- **Preprocessing**: Wavelet denoising (db4 DWT)
- **Feature Engineering**:
  - N-BEATS-style basis decomposition (trend + seasonality)
  - xLSTM-TS exponential gating features
  - PSX-specific seasonal patterns (Ramadan, EID, fiscal year)
  - 70+ technical indicators
- **Models**: 6-model ensemble
  - RandomForest (500 estimators)
  - ExtraTrees (500 estimators)
  - GradientBoosting (500 estimators)
  - XGBoost
  - LightGBM
  - Ridge Regression
- **Validation**: 5-fold walk-forward time series split
- **Output**: Daily predictions through December 2026

## Deployment

### Prerequisites
- GitHub account
- Vercel account
- E2B account (https://e2b.dev)
- Groq account (https://groq.com) - optional, for sentiment analysis

### Environment Variables

Set these in Vercel Dashboard > Project Settings > Environment Variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `E2B_API_KEY` | E2B API key for sandbox execution | Yes |
| `GROQ_API_KEY` | Groq API key for sentiment analysis | No |

### Deploy to Vercel

1. Fork or clone this repository
2. Push to your GitHub account
3. Import project in Vercel Dashboard
4. Set environment variables
5. Deploy

```bash
# Or deploy via CLI
npm i -g vercel
vercel --prod
```

## Project Structure

```
psx-prediction-app/
├── api/                          # Vercel serverless functions
│   ├── analyze.py               # Main analysis endpoint (E2B)
│   ├── screener.py              # Stock screener endpoint (E2B)
│   ├── sentiment.py             # Sentiment analysis (Groq)
│   └── health.py                # Health check endpoint
├── public/                       # Static frontend files
│   └── index.html               # Main application UI
├── e2b_scripts/                  # Scripts executed in E2B sandbox
│   ├── stock_analyzer.py        # Full ML pipeline
│   └── requirements.txt         # E2B sandbox dependencies
├── vercel.json                   # Vercel configuration
├── requirements.txt              # Vercel API dependencies
└── .env.example                  # Environment variables template
```

## API Endpoints

### POST /api/analyze
Runs full stock analysis in E2B sandbox.

**Request:**
```json
{
  "symbol": "LUCK"
}
```

**Response:**
```json
{
  "status": "complete",
  "symbol": "LUCK",
  "current_price": 850.00,
  "model_performance": {
    "r2": 0.95,
    "trend_accuracy": 0.72,
    "mase": 0.45
  },
  "daily_predictions": [...],
  "historical_data": [...]
}
```

### GET /api/screener
Returns top performing stocks based on technical indicators.

**Response:**
```json
{
  "success": true,
  "top_picks": [
    {
      "symbol": "LUCK",
      "current_price": 850.00,
      "return_1w": 2.5,
      "return_1m": 8.3,
      "signal": "BUY"
    }
  ]
}
```

### GET /api/sentiment/{symbol}
Returns AI-powered sentiment analysis using Groq LLM.

**Response:**
```json
{
  "success": true,
  "sentiment": {
    "symbol": "LUCK",
    "signal": "BULLISH",
    "sentiment_score": 0.65,
    "summary": "..."
  }
}
```

### GET /api/health
Returns API health status.

## Model Performance

Typical metrics on PSX stocks:
- **R2 Score**: 0.90-0.97
- **Trend Accuracy**: 65-75%
- **MASE**: < 1.0 (better than naive forecast)

## Limitations

- Analysis takes 30-60 seconds on first run (E2B cold start + model training)
- Predictions are for informational purposes only
- Past performance does not guarantee future results
- E2B sandbox has 5-minute execution limit per request

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires E2B_API_KEY in .env)
vercel dev
```

### Testing E2B Script Locally

```bash
cd e2b_scripts
pip install -r requirements.txt
python stock_analyzer.py LUCK
```

## License

MIT License. See LICENSE file for details.

## Disclaimer

This software is for educational and informational purposes only. It does not constitute financial advice. Stock market investments carry risk. Always conduct your own research before making investment decisions.
