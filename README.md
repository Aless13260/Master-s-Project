# Finance Guidance Extraction Pipeline

An automated system for extracting financial guidance (revenue forecasts, EPS estimates, margin targets, etc.) from SEC filings and corporate press releases using LLM-powered extraction.

## Overview

This project implements a complete pipeline for:
1. **Ingesting** financial documents from RSS feeds (SEC EDGAR, corporate investor relations)
2. **Parsing** HTML content from 8-K filings and press releases
3. **Filtering** documents likely to contain forward-looking guidance
4. **Extracting** structured guidance data using LLM (DeepSeek, OpenAI, GitHub Models)
5. **Storing** results in SQLite for analysis

## Quick Start

```bash
# 1. Clone and setup environment
git clone <repo-url>
cd FinanceProject
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys (at minimum: DEEPSEEK_API_KEY)

# 4. Run the pipeline
python run_pipeline.py
```

## Pipeline Modes

```bash
# Standard mode (fast, uses regex for period normalization)
python run_pipeline.py

# Enhanced mode (slower, uses agentic LLM for period normalization)
python run_pipeline.py --enhanced

# Skip certain stages
python run_pipeline.py --skip-ingest  # Skip RSS/HTML fetching
python run_pipeline.py --skip-extract # Skip LLM extraction

# Test with limited items
python run_pipeline.py --max-items 10
```

## Project Structure

```
├── run_pipeline.py          # Main orchestrator
├── config.py                # Centralized configuration
├── sources.yaml             # RSS feed definitions
├── requirements.txt         # Python dependencies
├── .env.example             # Template for API keys
│
├── ingestion_scripts/       # Data fetching
│   ├── rss_guidance_ingest.py   # RSS feed polling
│   └── web_parse_trafilatura.py # HTML content extraction
│
├── extractor_lib/           # Core extraction logic
│   ├── LLM_extractor.py     # LLM-based guidance extraction
│   ├── regex_filter.py      # Pre-filter candidates
│   ├── period_normalizer.py # Fiscal period normalization
│   └── guidance_schema.py   # Pydantic models
│
├── evaluation/              # Model evaluation
│   ├── run_extraction_on_gt.py  # Run on ground truth
│   ├── evaluate_extraction.py   # Compute metrics
│   └── ground_truth/            # Labeled test data
│
└── migrate_to_sqlite.py     # Export to SQLite database
```

## Configuration

Set these environment variables in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes* | DeepSeek API key (default provider) |
| `OPENAI_API_KEY` | No | OpenAI API key (for --provider openai) |
| `GITHUB_MODELS_API_KEY` | No | GitHub Models API key |
| `CONTACT_EMAIL` | Recommended | Email for User-Agent (SEC compliance) |

*At least one LLM API key is required.

## Evaluation

```bash
# Run extraction on ground truth data
python evaluation/run_extraction_on_gt.py

# With enhanced mode
python evaluation/run_extraction_on_gt.py --enhanced

# Compute metrics
python evaluation/evaluate_extraction.py
```

## See Also

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Detailed reproducibility guide
- [sources.yaml](sources.yaml) - RSS feed configuration