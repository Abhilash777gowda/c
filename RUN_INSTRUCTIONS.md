# How to Run CRIMSON-India

## Prerequisites
- Python 3.11+ (already installed)
- Virtual environment already set up in `venv/`

## Quick Start

### Step 1: Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### Step 2: Install Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Code

#### Option A: Run Main Pipeline (Recommended for first time)
```powershell
# Quick test with synthetic data (no internet required)
python main.py --use-synthetic

# Or run with real scraping (requires internet)
python main.py
```

#### Option B: Run Streamlit Dashboard (Interactive Web Interface)
```powershell
streamlit run app.py
```
This will open a web browser automatically at `http://localhost:8501`

## Command Line Options for main.py

- `--use-synthetic` - Use synthetic data (bypasses web scraping, faster for testing)
- `--use-rss` - Use live RSS feeds for real-time data
- `--skip-muril` - Skip MuRIL training (saves time if no Hindi data)
- `--skip-xlmr` - Skip XLM-RoBERTa training (faster runs)
- `--epochs N` - Number of training epochs (default: 2)

## Examples

```powershell
# Quick test run
python main.py --use-synthetic

# Full pipeline with real data
python main.py

# Fast run (skip some models)
python main.py --use-synthetic --skip-muril --skip-xlmr

# Custom epochs
python main.py --use-synthetic --epochs 3
```

## Troubleshooting

1. **If activation script fails**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell as Administrator
2. **If dependencies missing**: Run `pip install -r requirements.txt`
3. **If models fail to download**: Check internet connection (transformers will download models automatically)
