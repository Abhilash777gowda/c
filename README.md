# CRIMSON-India

A Deep Learning Framework for Real-time Crime and Accident Monitoring from Indian Online News.

## Overview
This automated pipeline collects online Indian news articles, processes them using NLP, classifies them into multiple crime categories (Theft, Assault, Accident, Drug Crime, Cybercrime, Non-Crime) using deep learning transformer models, and analyzes crime trends over time.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline (with Synthetic Data to test end-to-end without scraping blocks):**
   ```bash
   python main.py --use-synthetic
   ```

3. **Run Pipeline (Real Scraping Mode):**
   ```bash
   python main.py
   ```

## Directory Structure
- `scraper/`: Web scraping scripts.
- `preprocessing/`: Text cleaning and normalisation.
- `utils/`: Helpers, PyTorch dataset classes, synthetic data generator.
- `models/`: TF-IDF baseline, BiLSTM, and HuggingFace Transformers logic.
- `analysis/`: Plotting and NCRB correlation validation.
- `plots/`: Generated output graphs.
- `data/`: Datasets.
