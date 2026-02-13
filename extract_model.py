
import os
import shutil
from pathlib import Path

# Source and destination
SOURCE_DIR = Path(".")
DEST_DIR = Path("standalone_model")

# Files to extract (relative to source/backend)
BACKEND_FILES = [
    "stock_analyzer_fixed.py",
    "research_model.py",
    "sota_model.py",
    "williams_r_classifier.py",
    "sector_models.py",
    "stacking_ensemble.py",
    "prediction_stability.py",
    "prediction_reasoning.py",
    "monthly_forecast.py",
    "prediction_logger.py",
    "external_features.py",
    "validated_indicators.py",
    "feature_validation.py",
    "kse100_analyzer.py",
    "sentiment_analyzer.py",
    "sentiment_math.py",
    "enhanced_news_fetcher.py",
    "article_scraper.py",
    "tradingview_scraper.py",
    "brecorder_scraper.py",
    "__init__.py"
]

def extract_files():
    print(f"🚀 Starting extraction to {DEST_DIR.absolute()}")
    
    # Ensure backend dir exists
    (DEST_DIR / "backend").mkdir(parents=True, exist_ok=True)
    
    # Copy backend files
    success_count = 0
    for filename in BACKEND_FILES:
        src = SOURCE_DIR / "backend" / filename
        dst = DEST_DIR / "backend" / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✅ Copied: {filename}")
            success_count += 1
        else:
            print(f"⚠️ Missing: {filename}")
            
    # Copy models directory if it exists and has content
    src_models = SOURCE_DIR / "backend" / "models"
    if src_models.exists():
        dst_models = DEST_DIR / "backend" / "models"
        if dst_models.exists():
            shutil.rmtree(dst_models)
        shutil.copytree(src_models, dst_models)
        print(f"✅ Copied: backend/models directory")
    
    print(f"\n🎉 Extraction complete! Copied {success_count} files.")

if __name__ == "__main__":
    extract_files()
