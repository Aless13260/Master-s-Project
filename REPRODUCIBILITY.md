# Reproducibility Guide

To ensure this project is reproducible, follow these steps:

## 1. Environment Setup
- Use Python 3.10+
- Create a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/Mac
  .venv\Scripts\activate     # Windows
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 2. Configuration
- Ensure `.env` contains necessary API keys (e.g., `DEEPSEEK_API_KEY`).
- `sources.yaml` defines the input data. Keep a versioned copy of this file.

## 3. Data Pipeline
- The pipeline is orchestrated by `run_pipeline.py`.
- Run the full pipeline:
  ```bash
  python run_pipeline.py
  ```
- Intermediate files are stored in `ingestion_json/` and `extractor_lib/`.
- Final database is `finance_data.db`.

## 4. Determinism
- LLM outputs can vary. `llm_setup.py` defaults to `temperature=0.0` for more deterministic results.
- Random seeds are not explicitly set for network operations, so retry logic handles transient failures.

## 5. Docker (Optional)
For maximum reproducibility, use a Docker container.
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "run_pipeline.py"]
```
