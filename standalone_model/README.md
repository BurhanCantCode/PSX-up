# PSX Prediction Model (Standalone)

This directory contains the standalone core prediction model extracted from the main application. It is designed to run independently for 21-day PSX stock predictions.

## 📂 Structure

- `backend/`: Core model logic and feature engineering.
- `data/`: Directory for logs, cache, and state files.
- `requirements.txt`: Python dependencies.

## 🚀 Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file in this directory if you plan to use sentiment analysis or premium features:
    ```ini
    GROQ_API_KEY=your_groq_api_key_here
    # Add other keys as needed
    ```

## 🏃‍♂️ Usage

To run the model, you can import `StockAnalyzer` from `backend.stock_analyzer_fixed` or run it directly if a main block exists.

Example usage script (`run_prediction.py`):
```python
import sys
from pathlib import Path

# Ensure backend module can be found
sys.path.append(str(Path(__file__).parent))

from backend.stock_analyzer_fixed import StockAnalyzer

def main():
    analyzer = StockAnalyzer()
    symbol = "OGDC"  # Example symbol
    
    print(f"Running prediction for {symbol}...")
    try:
        prediction = analyzer.predict_stock(symbol, days=21)
        print("Prediction Result:", prediction)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## 📦 extracted files

The following files were extracted from the main application:
- `backend/stock_analyzer_fixed.py`
- `backend/research_model.py`
- and other core dependencies.
