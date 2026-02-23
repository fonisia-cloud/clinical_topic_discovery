# Clinical Topic Discovery (V1)

A lightweight Streamlit app for clinical topic discovery and evidence-backed topic planning.

## What V1 does

- Search PubMed with keyword + MeSH + date range + study type filters
- Keyword/MeSH inputs support boolean syntax (`AND`/`OR`/`NOT`); comma remains OR for backward compatibility
- Keyword mode supports `PubMed default` (closer to website behavior) and `Title/Abstract only`
- Build a structured local evidence library (SQLite)
- Generate evidence landscape views (disease/intervention/outcome/study-design signals)
- Surface trend insights and emerging terms
- Manage a manual included-paper set directly in V1
- Generate final topic ideas with supporting PMIDs
- Optionally enhance topic ideas with OpenAI-compatible LLM while keeping evidence-linked output
- Sort results by publication year, citation count, JCR IF, CAS major tier, or journal impact proxy

## Project structure

```text
clinical_topic_discovery/
  app.py
  requirements.txt
  core/
    analyzer.py
    jcr_integration.py
    llm_writer.py
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

## Notes

- Optional: add your NCBI email/api_key in the sidebar for better API limits.
- In `Search Results`, use `Fetch JCR IF` / `Fetch CAS partition` to integrate ShowJCR database (auto-download to `data/showjcr_jcr.db`).
- In `Topic Ideas`, you can export evidence package JSONL and call LLM enhancement in-page.
- Data is stored locally in `data/app.db`.
