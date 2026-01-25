"""
📊 PREDICTION LOGGER
Tracks predictions vs actual outcomes to measure real-world accuracy.

Features:
- Logs every prediction with timestamp
- Records actual outcomes when available
- Calculates rolling accuracy metrics
- Exports to JSON for analysis
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Log file location
LOG_DIR = Path(__file__).parent.parent / "data" / "prediction_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class PredictionLogger:
    """Track predictions and measure accuracy over time."""

    def __init__(self):
        self.log_file = LOG_DIR / "prediction_log.json"
        self.predictions = self._load_log()

    def _load_log(self) -> List[Dict]:
        """Load existing prediction log."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load prediction log: {e}")
        return []

    def _save_log(self):
        """Save prediction log to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save prediction log: {e}")

    def log_prediction(
        self,
        symbol: str,
        current_price: float,
        predicted_price: float,
        predicted_direction: str,
        confidence: float,
        horizon_days: int = 1,
        williams_signal: Optional[str] = None,
        sector: Optional[str] = None
    ) -> Dict:
        """
        Log a new prediction.

        Args:
            symbol: Stock symbol
            current_price: Current price at prediction time
            predicted_price: Predicted future price
            predicted_direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            confidence: Model confidence (0-1)
            horizon_days: Days until prediction should be evaluated
            williams_signal: Williams %R classifier signal (if available)
            sector: Detected sector (if available)

        Returns:
            The logged prediction entry
        """
        prediction_date = datetime.now()
        evaluation_date = prediction_date + timedelta(days=horizon_days)

        entry = {
            'id': f"{symbol}_{prediction_date.strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'prediction_date': prediction_date.isoformat(),
            'evaluation_date': evaluation_date.strftime('%Y-%m-%d'),
            'horizon_days': horizon_days,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'predicted_change_pct': round((predicted_price - current_price) / current_price * 100, 2),
            'predicted_direction': predicted_direction,
            'confidence': round(confidence, 2),
            'williams_signal': williams_signal,
            'sector': sector,
            # To be filled later
            'actual_price': None,
            'actual_change_pct': None,
            'actual_direction': None,
            'direction_correct': None,
            'evaluated': False
        }

        self.predictions.append(entry)
        self._save_log()

        print(f"📝 Logged prediction: {symbol} {predicted_direction} ({confidence:.0%} conf)")
        return entry

    def update_actual(self, symbol: str, evaluation_date: str, actual_price: float) -> Optional[Dict]:
        """
        Update a prediction with actual outcome.

        Args:
            symbol: Stock symbol
            evaluation_date: Date to evaluate (YYYY-MM-DD)
            actual_price: Actual price on evaluation date

        Returns:
            Updated prediction entry or None if not found
        """
        for pred in self.predictions:
            if (pred['symbol'] == symbol and
                pred['evaluation_date'] == evaluation_date and
                not pred['evaluated']):

                current_price = pred['current_price']
                actual_change = (actual_price - current_price) / current_price * 100

                pred['actual_price'] = round(actual_price, 2)
                pred['actual_change_pct'] = round(actual_change, 2)
                pred['actual_direction'] = 'BULLISH' if actual_change > 0 else 'BEARISH'

                # Check if direction was correct
                pred_dir = pred['predicted_direction']
                actual_dir = pred['actual_direction']
                pred['direction_correct'] = (
                    (pred_dir == 'BULLISH' and actual_dir == 'BULLISH') or
                    (pred_dir == 'BEARISH' and actual_dir == 'BEARISH') or
                    (pred_dir == 'NEUTRAL')  # Neutral is always "correct" (conservative)
                )

                pred['evaluated'] = True
                self._save_log()

                status = "✅ CORRECT" if pred['direction_correct'] else "❌ WRONG"
                print(f"📊 Evaluated {symbol}: Predicted {pred_dir}, Actual {actual_dir} → {status}")

                return pred

        return None

    def get_accuracy_stats(self, symbol: Optional[str] = None, days: int = 30) -> Dict:
        """
        Calculate accuracy statistics.

        Args:
            symbol: Filter by symbol (None = all)
            days: Look back period in days

        Returns:
            Dictionary with accuracy metrics
        """
        cutoff = datetime.now() - timedelta(days=days)

        evaluated = [
            p for p in self.predictions
            if p['evaluated'] and
            datetime.fromisoformat(p['prediction_date']) > cutoff and
            (symbol is None or p['symbol'] == symbol)
        ]

        if not evaluated:
            return {
                'total_predictions': 0,
                'direction_accuracy': None,
                'avg_confidence': None,
                'message': 'No evaluated predictions in period'
            }

        correct = sum(1 for p in evaluated if p['direction_correct'])
        total = len(evaluated)

        # Average error
        errors = [
            abs(p['predicted_change_pct'] - p['actual_change_pct'])
            for p in evaluated
            if p['actual_change_pct'] is not None
        ]

        # Confidence vs accuracy correlation
        high_conf = [p for p in evaluated if p['confidence'] > 0.7]
        high_conf_correct = sum(1 for p in high_conf if p['direction_correct'])

        return {
            'total_predictions': total,
            'direction_accuracy': round(correct / total * 100, 1) if total > 0 else None,
            'avg_confidence': round(np.mean([p['confidence'] for p in evaluated]) * 100, 1),
            'avg_error_pct': round(np.mean(errors), 2) if errors else None,
            'high_confidence_accuracy': round(high_conf_correct / len(high_conf) * 100, 1) if high_conf else None,
            'period_days': days,
            'symbol_filter': symbol
        }

    def get_pending_evaluations(self) -> List[Dict]:
        """Get predictions that need actual price updates."""
        today = datetime.now().strftime('%Y-%m-%d')

        pending = [
            p for p in self.predictions
            if not p['evaluated'] and p['evaluation_date'] <= today
        ]

        return pending

    def get_recent_predictions(self, limit: int = 10, symbol: Optional[str] = None) -> List[Dict]:
        """Get most recent predictions."""
        filtered = [
            p for p in self.predictions
            if symbol is None or p['symbol'] == symbol
        ]

        return sorted(
            filtered,
            key=lambda x: x['prediction_date'],
            reverse=True
        )[:limit]

    def export_to_csv(self, filepath: str = None) -> str:
        """Export prediction log to CSV for analysis."""
        import csv

        if filepath is None:
            filepath = LOG_DIR / f"predictions_export_{datetime.now().strftime('%Y%m%d')}.csv"

        fieldnames = [
            'id', 'symbol', 'prediction_date', 'evaluation_date', 'horizon_days',
            'current_price', 'predicted_price', 'predicted_change_pct', 'predicted_direction',
            'confidence', 'williams_signal', 'sector',
            'actual_price', 'actual_change_pct', 'actual_direction', 'direction_correct', 'evaluated'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.predictions)

        print(f"📁 Exported {len(self.predictions)} predictions to {filepath}")
        return str(filepath)


# Singleton instance
_logger = None


def get_prediction_logger() -> PredictionLogger:
    """Get the singleton prediction logger instance."""
    global _logger
    if _logger is None:
        _logger = PredictionLogger()
    return _logger


# ============================================================================
# CLI for checking accuracy
# ============================================================================

if __name__ == "__main__":
    import sys

    logger = get_prediction_logger()

    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        # Show accuracy stats
        symbol = sys.argv[2] if len(sys.argv) > 2 else None
        stats = logger.get_accuracy_stats(symbol=symbol)

        print("\n" + "=" * 60)
        print("📊 PREDICTION ACCURACY STATS")
        print("=" * 60)

        if stats['total_predictions'] == 0:
            print("\n   No evaluated predictions yet.")
        else:
            print(f"\n   Total Predictions: {stats['total_predictions']}")
            print(f"   Direction Accuracy: {stats['direction_accuracy']}%")
            print(f"   Avg Confidence: {stats['avg_confidence']}%")
            print(f"   Avg Error: {stats['avg_error_pct']}%")
            if stats['high_confidence_accuracy']:
                print(f"   High-Conf Accuracy: {stats['high_confidence_accuracy']}%")

    elif len(sys.argv) > 1 and sys.argv[1] == 'pending':
        # Show pending evaluations
        pending = logger.get_pending_evaluations()

        print("\n" + "=" * 60)
        print("⏳ PENDING EVALUATIONS")
        print("=" * 60)

        if not pending:
            print("\n   No pending evaluations.")
        else:
            for p in pending:
                print(f"\n   {p['symbol']}: {p['predicted_direction']} ({p['confidence']:.0%})")
                print(f"      Predicted: {p['predicted_change_pct']:+.1f}%")
                print(f"      Evaluate on: {p['evaluation_date']}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'export':
        # Export to CSV
        logger.export_to_csv()

    else:
        print("\nUsage:")
        print("  python prediction_logger.py stats [symbol]  - Show accuracy stats")
        print("  python prediction_logger.py pending         - Show pending evaluations")
        print("  python prediction_logger.py export          - Export to CSV")
