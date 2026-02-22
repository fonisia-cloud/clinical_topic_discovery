# Clinical Topic Discovery (V1)

A lightweight Streamlit app for early-stage clinical topic discovery before writing an original paper.

## What V1 does

- Search PubMed with keyword + MeSH + date range + study type filters
- Keyword/MeSH inputs support boolean syntax (`AND`/`OR`/`NOT`); comma remains OR for backward compatibility
- Keyword mode supports `PubMed default` (closer to website behavior) and `Title/Abstract only`
- Build a structured local evidence library (SQLite)
- Generate evidence landscape views (disease/intervention/outcome/study-design signals)
- Surface trend insights and emerging terms
- Produce candidate topic ideas with supporting PMIDs
- Re-rank topic ideas with configurable weights (clinical value / innovation / feasibility)
- Sort results by publication year, citation count, JCR IF, CAS major tier, or journal impact proxy
- View full abstract from the built-in abstract viewer

## Project structure

```text
clinical_topic_discovery/
  app.py
  requirements.txt
  core/
    analyzer.py
    pubmed_client.py
    query_builder.py
    repository.py
    topic_generator.py
  data/
    app.db   # created automatically on first run
  docs/
    USER_GUIDE_CN.md
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Documentation

- Chinese user guide: `docs/USER_GUIDE_CN.md`

## Notes

- Optional: add your NCBI email/api_key in the sidebar for better API limits.
- In `Search Results`, use `Fetch JCR IF` / `Fetch CAS partition` to integrate ShowJCR database (auto-download to `data/showjcr_jcr.db`).
- Topic scoring weights are adjusted in the `Topic Ideas` tab (for the selected run).
- V1 focuses on discovery and evidence synthesis, not manuscript drafting yet.
- Data is stored locally in `data/app.db`.
