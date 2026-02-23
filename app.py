from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.analyzer import compute_trends, evidence_map
from core.jcr_integration import (
    ensure_showjcr_db,
    load_cas_index,
    load_jcr_index,
    match_papers_with_cas,
    match_papers_with_jcr,
)
from core.llm_writer import call_openai_compatible, enhance_topic_ideas_with_llm
from core.pubmed_client import PubMedClient
from core.query_builder import ARTICLE_TYPE_MAP, build_query
from core.repository import Repository
from core.topic_generator import DEFAULT_SCORE_WEIGHTS, generate_topic_candidates

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
DB_PATH = DATA_DIR / "app.db"
SHOWJCR_DB_PATH = DATA_DIR / "showjcr_jcr.db"

@st.cache_resource
def get_repo() -> Repository:
    return Repository(str(DB_PATH))


def render_metric_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", len(df))
    c2.metric("Unique Journals", int(df["journal"].nunique()))

    year_series = pd.to_numeric(df["pub_year"], errors="coerce").dropna()
    c3.metric("Median Year", int(year_series.median()) if not year_series.empty else "-")

    with_doi = int(df["doi"].fillna("").astype(str).str.len().gt(0).sum())
    c4.metric("With DOI", with_doi)


def bar_chart(df: pd.DataFrame, title: str, top_n: int = 12) -> None:
    if df.empty:
        st.info(f"No data for {title}.")
        return

    draw = df.sort_values("count", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(draw["item"], draw["count"], color="#2f6b8a")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)


def enrich_run_metrics(repo: Repository, run_df: pd.DataFrame) -> int:
    if run_df.empty:
        return 0

    client = PubMedClient()
    rows = run_df[["pmid", "doi"]].fillna("").to_dict(orient="records")
    metrics = client.fetch_openalex_metrics(rows)
    repo.update_paper_metrics(metrics)
    return len(metrics)


def enrich_run_jcr(repo: Repository, run_df: pd.DataFrame, table_name: str = "JCR2024") -> int:
    if run_df.empty:
        return 0

    db_path = ensure_showjcr_db(str(SHOWJCR_DB_PATH))
    jcr_index = load_jcr_index(str(db_path), table_name=table_name)
    paper_rows = run_df[["pmid", "journal"]].fillna("").to_dict(orient="records")
    metrics = match_papers_with_jcr(paper_rows, jcr_index)
    repo.update_paper_jcr(metrics)
    return len(metrics)


def enrich_run_cas(repo: Repository, run_df: pd.DataFrame, table_name: str = "FQBJCR2025") -> int:
    if run_df.empty:
        return 0

    db_path = ensure_showjcr_db(str(SHOWJCR_DB_PATH))
    cas_index = load_cas_index(str(db_path), table_name=table_name)
    paper_rows = run_df[["pmid", "journal"]].fillna("").to_dict(orient="records")
    metrics = match_papers_with_cas(paper_rows, cas_index)
    repo.update_paper_cas(metrics)
    return len(metrics)


def run_search(repo: Repository, score_weights: dict[str, float]) -> int | None:
    st.sidebar.subheader("Search Settings")
    project_name = st.sidebar.text_input("Project name", value="Clinical Topic Discovery")
    keywords = st.sidebar.text_area(
        "Keywords (supports AND/OR/NOT; comma = OR)",
        value="",
        placeholder='e.g. (large language model AND urology) OR prostate cancer',
        height=90,
    )
    mesh_terms = st.sidebar.text_input(
        "MeSH terms (optional, supports AND/OR/NOT; comma = OR)",
        value="",
        placeholder='e.g. "Kidney Diseases" AND "Sodium-Glucose Transporter 2 Inhibitors"',
    )

    keyword_mode = st.sidebar.selectbox(
        "Keyword matching mode",
        options=["PubMed default (recommended)", "Title/Abstract only"],
        index=0,
        help="PubMed default behaves closer to the website search box (Automatic Term Mapping).",
    )

    col1, col2 = st.sidebar.columns(2)
    start_year = col1.number_input("Start year", min_value=1980, max_value=2100, value=2018)
    end_year = col2.number_input("End year", min_value=1980, max_value=2100, value=2026)

    selected_types = st.sidebar.multiselect(
        "Study types",
        options=list(ARTICLE_TYPE_MAP.keys()),
        default=["Systematic Review", "Meta-Analysis", "Randomized Controlled Trial"],
    )

    max_results = st.sidebar.slider("Max papers", min_value=50, max_value=500, value=250, step=25)

    st.sidebar.markdown("---")
    email = st.sidebar.text_input("NCBI email (optional)")
    api_key = st.sidebar.text_input("NCBI api_key (optional)", type="password")

    search_clicked = st.sidebar.button("Run Search and Analyze", type="primary")

    if not search_clicked:
        return None

    if not keywords.strip():
        st.error("Please enter at least one keyword.")
        return None

    query = build_query(
        keywords=keywords,
        mesh_terms=mesh_terms,
        start_year=int(start_year),
        end_year=int(end_year),
        article_types=selected_types,
        keyword_mode=("pubmed_default" if keyword_mode.startswith("PubMed default") else "field_restricted"),
    )

    client = PubMedClient(email=email.strip(), api_key=api_key.strip())

    with st.spinner("Searching PubMed and preparing analysis..."):
        search_result = client.search_pmids(query=query, retmax=int(max_results))
        pmids = search_result.get("idlist", [])
        total_found = int(search_result.get("count", 0))

        if not pmids:
            st.warning("No records found. Try broader keywords or fewer filters.")
            return None

        summaries = client.fetch_summaries(pmids)
        abstracts = client.fetch_abstract_metadata(pmids)
        records = client.merge_records(pmids, summaries, abstracts)

        run_id = repo.create_run(
            project_name=project_name.strip(),
            raw_query=keywords.strip(),
            expanded_query=query,
            start_year=int(start_year),
            end_year=int(end_year),
            article_types=selected_types,
            max_results=int(max_results),
            total_results=total_found,
            score_weights=score_weights,
        )

        repo.upsert_papers(records)
        repo.link_run_papers(run_id, [r["pmid"] for r in records])

        df = repo.get_run_papers(run_id)
        topics = generate_topic_candidates(
            df,
            top_n=5,
            score_weights=score_weights,
            query_text=keywords,
        )
        repo.save_topic_candidates(run_id, topics)

    st.success(
        f"Completed run #{run_id}. PubMed matched {total_found} papers; "
        f"this app retrieved top {len(records)} (retmax={max_results})."
    )
    return run_id


def _format_created_at_local(created_at: str) -> str:
    parsed = pd.to_datetime(created_at, errors="coerce", utc=True)
    if pd.isna(parsed):
        return str(created_at)

    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is None:
        return str(parsed.strftime("%Y-%m-%d %H:%M:%S UTC"))

    return str(parsed.tz_convert(local_tz).strftime("%Y-%m-%d %H:%M:%S"))


def _translate_abstract_to_zh(
    abstract_text: str,
    api_base: str,
    api_key: str,
    model_name: str,
) -> str:
    prompt = (
        "Translate the following medical abstract into professional academic Chinese. "
        "Keep all facts, numbers, p-values, abbreviations, and drug names accurate. "
        "Do not add, remove, or infer information beyond the original text.\n\n"
        f"Abstract:\n{abstract_text}"
    )
    return call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        temperature=0.0,
        timeout=180,
    )


def _build_topic_followup_prompt(
    topic_title: str,
    rationale: str,
    key_gap: str,
    supporting_pmids: list[str],
    evidence_records: list[dict],
    requested_records: list[dict],
    history_pairs: list[tuple[str, str]],
    user_message: str,
) -> str:
    requested_lines = []
    for row in requested_records[:10]:
        requested_lines.append(
            f"PMID {row.get('pmid')}: {row.get('title')} ({row.get('pub_year')}, {row.get('journal')}); "
            f"type={row.get('publication_types', '')}; full_abstract={str(row.get('abstract', ''))[:2500]}"
        )

    evidence_lines = []
    for row in evidence_records[:40]:
        evidence_lines.append(
            f"PMID {row.get('pmid')}: {row.get('title')} ({row.get('pub_year')}, {row.get('journal')}); "
            f"type={row.get('publication_types', '')}; abstract={str(row.get('abstract', ''))[:900]}"
        )

    history_lines = []
    for q, a in history_pairs[-6:]:
        history_lines.append(f"User: {q}")
        history_lines.append(f"Assistant: {a}")

    return (
        "You are a rigorous medical research advisor.\n"
        "Use only provided evidence. Do not fabricate studies or PMIDs.\n"
        "If user names a PMID and it is provided in requested PMID records, explicitly use that record first.\n"
        "When giving recommendations, clearly indicate uncertainty and next validation actions.\n"
        "Keep response concise and practical for study planning.\n\n"
        f"Current topic: {topic_title}\n"
        f"Rationale: {rationale}\n"
        f"Gap: {key_gap}\n"
        f"Supporting PMIDs: {', '.join(supporting_pmids)}\n\n"
        "Requested PMID records (highest priority):\n"
        + "\n".join(requested_lines)
        + "\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
        + "\n\nPrevious discussion:\n"
        + "\n".join(history_lines)
        + "\n\nUser question:\n"
        + user_message
    )


def _normalize_pmid(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _extract_pmids_from_text(text: str) -> list[str]:
    import re

    found = re.findall(r"\b\d{7,9}\b", text or "")
    deduped = []
    seen = set()
    for pmid in found:
        if pmid in seen:
            continue
        seen.add(pmid)
        deduped.append(pmid)
    return deduped


def _load_default_llm_session(repo: Repository) -> None:
    if st.session_state.get("llm_profile_initialized"):
        return

    default_profile = repo.get_default_llm_profile()
    if default_profile:
        st.session_state["v1_ti_api_base"] = str(default_profile.get("api_base", ""))
        st.session_state["v1_ti_api_key"] = str(default_profile.get("api_key", ""))
        st.session_state["v1_ti_model_name"] = str(default_profile.get("model_name", ""))
        st.session_state["v1_ti_profile_id"] = int(default_profile.get("id", 0) or 0)
        st.session_state["v1_ti_profile_name"] = str(default_profile.get("profile_name", ""))
    st.session_state["llm_profile_initialized"] = True


def main() -> None:
    st.set_page_config(page_title="Clinical Topic Discovery", page_icon="🩺", layout="wide")
    st.title("Clinical Topic Discovery")
    st.caption("V1: PubMed discovery + evidence-backed topic ideas")

    repo = get_repo()
    _load_default_llm_session(repo)
    default_search_weights = dict(DEFAULT_SCORE_WEIGHTS)

    new_run_id = run_search(repo, score_weights=default_search_weights)

    runs = repo.get_runs()
    if runs.empty:
        st.info("No run yet. Use the sidebar to start your first PubMed analysis.")
        return

    runs = runs.copy()
    runs["created_at_local"] = runs["created_at"].astype(str).apply(_format_created_at_local)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Run Context")
    with st.sidebar.expander("History Management", expanded=False):
        confirm_clear = st.checkbox("Confirm clear all history", key="sb_confirm_clear_all")
        if st.button("Clear all history", key="sb_clear_all_history_btn"):
            if confirm_clear:
                repo.clear_history()
                st.success("All history cleared.")
                st.rerun()
            else:
                st.warning("Please confirm before clearing all history.")

    filter_text = st.sidebar.text_input("Filter runs", value="", placeholder="project/query", key="sb_filter_runs")
    run_view = runs
    if filter_text.strip():
        mask = (
            runs["project_name"].fillna("").str.contains(filter_text, case=False, na=False)
            | runs["expanded_query"].fillna("").str.contains(filter_text, case=False, na=False)
        )
        run_view = runs[mask]

    if run_view.empty:
        st.warning("No history matches current filter.")
        return

    show_limit = st.sidebar.number_input(
        "Show recent", min_value=1, max_value=max(1, len(run_view)), value=min(40, len(run_view)), step=1
    )
    run_view = run_view.head(int(show_limit))

    run_options = run_view["id"].tolist()
    default_idx = run_options.index(new_run_id) if new_run_id in run_options else 0

    selected_run_id = st.sidebar.selectbox(
        "Select run",
        options=run_options,
        index=default_idx,
        key="sb_select_run",
        format_func=lambda rid: (
            f"#{rid} | {run_view.loc[run_view['id'] == rid, 'project_name'].iloc[0]} | "
            f"{run_view.loc[run_view['id'] == rid, 'created_at_local'].iloc[0]}"
        ),
    )

    delete_confirm = st.sidebar.checkbox("Confirm delete selected run", key=f"sb_confirm_delete_{selected_run_id}")
    if st.sidebar.button("Delete selected run", key=f"sb_delete_run_btn_{selected_run_id}"):
        if delete_confirm:
            repo.delete_run(int(selected_run_id))
            st.success(f"Run #{selected_run_id} deleted.")
            st.rerun()
        else:
            st.warning("Please confirm deletion first.")

    run_info = runs[runs["id"] == selected_run_id].iloc[0]
    st.markdown(f"**Current Run**: `#{selected_run_id}` | **Project**: `{run_info['project_name']}`")
    st.markdown(f"**Query**: `{run_info['expanded_query']}`")

    df = repo.get_run_papers(int(selected_run_id))
    if df.empty:
        st.warning("No papers in this run.")
        return

    render_metric_row(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Search Results", "Evidence Map", "Trend Insights", "Topic Ideas"])

    with tab1:
        st.subheader("Structured Paper List")

        metric_c1, metric_c2, metric_c3, metric_c4 = st.columns([1.1, 1.3, 1.3, 2.6])
        with metric_c1:
            if st.button("Fetch citation metrics", key=f"fetch_metrics_{selected_run_id}"):
                with st.spinner("Fetching citation data from OpenAlex..."):
                    updated = enrich_run_metrics(repo, df)
                st.success(f"Updated citation metrics for {updated} papers (DOI-matched).")
                st.rerun()
        with metric_c2:
            jcr_table = st.selectbox("JCR table", options=["JCR2024", "JCR2023", "JCR2022"], index=0, key=f"jcr_table_{selected_run_id}")
            if st.button("Fetch JCR IF", key=f"fetch_jcr_{selected_run_id}"):
                with st.spinner("Downloading/loading ShowJCR database and matching journals..."):
                    updated_jcr = enrich_run_jcr(repo, df, table_name=jcr_table)
                st.success(f"Updated JCR IF for {updated_jcr} papers (journal-matched).")
                st.rerun()
        with metric_c3:
            cas_table = st.selectbox("CAS table", options=["FQBJCR2025", "FQBJCR2023", "FQBJCR2022"], index=0, key=f"cas_table_{selected_run_id}")
            if st.button("Fetch CAS partition", key=f"fetch_cas_{selected_run_id}"):
                with st.spinner("Matching journals against ShowJCR CAS tables..."):
                    updated_cas = enrich_run_cas(repo, df, table_name=cas_table)
                st.success(f"Updated CAS partition for {updated_cas} papers (journal-matched).")
                st.rerun()
        with metric_c4:
            st.caption("Citation count from OpenAlex. JCR/CAS from ShowJCR local database.")

        inclusion_df = repo.get_inclusion_table(int(selected_run_id))
        inclusion_meta = inclusion_df[["pmid", "include_flag", "tags", "note"]].copy() if not inclusion_df.empty else pd.DataFrame(columns=["pmid", "include_flag", "tags", "note"])

        base_table = df.copy()
        base_table["pmid"] = base_table["pmid"].astype(str)
        if not inclusion_meta.empty:
            inclusion_meta["pmid"] = inclusion_meta["pmid"].astype(str)
            base_table = base_table.merge(inclusion_meta, on="pmid", how="left")
        else:
            base_table["include_flag"] = 0
            base_table["tags"] = ""
            base_table["note"] = ""

        base_table["include_flag"] = base_table["include_flag"].fillna(0).astype(int).astype(bool)
        base_table["tags"] = base_table["tags"].fillna("").astype(str)
        base_table["note"] = base_table["note"].fillna("").astype(str)

        years = sorted(pd.to_numeric(df["pub_year"], errors="coerce").dropna().astype(int).unique().tolist())
        selected_years = st.multiselect("Filter by year", options=years, default=years[-5:] if len(years) > 5 else years)

        filtered = base_table.copy()
        if selected_years:
            filtered = filtered[pd.to_numeric(filtered["pub_year"], errors="coerce").isin(selected_years)]

        c_sort1, c_sort2 = st.columns([2, 1])
        sort_option = c_sort1.selectbox(
            "Sort by",
            options=["Publication year", "Citation count", "JCR IF", "CAS major tier", "Journal impact proxy", "Title A-Z"],
            index=0,
        )
        descending = c_sort2.checkbox("Descending", value=True)

        sort_map = {
            "Publication year": "pub_year",
            "Citation count": "citation_count",
            "JCR IF": "jcr_if",
            "CAS major tier": "cas_major_tier",
            "Journal impact proxy": "journal_impact_score",
            "Title A-Z": "title",
        }
        sort_col = sort_map[sort_option]

        view_df = filtered.copy()
        if sort_col in {"pub_year", "citation_count", "jcr_if", "cas_major_tier", "journal_impact_score"}:
            view_df[sort_col] = pd.to_numeric(view_df[sort_col], errors="coerce")
            view_df = view_df.sort_values(sort_col, ascending=not descending, na_position="last")
        else:
            view_df = view_df.sort_values(sort_col, ascending=not descending, na_position="last")

        show_abstract_preview = st.checkbox("Show abstract preview in table", value=True)
        preview_len = st.slider("Preview length", min_value=120, max_value=800, value=280, step=20)

        view_df["pubmed_url"] = "https://pubmed.ncbi.nlm.nih.gov/" + view_df["pmid"].astype(str) + "/"
        if show_abstract_preview:
            view_df["abstract_preview"] = view_df["abstract"].fillna("").astype(str).str.slice(0, preview_len)
            show_cols = [
                "pmid",
                "title",
                "abstract_preview",
                "journal",
                "pub_year",
                "citation_count",
                "journal_impact_score",
                "jcr_if",
                "jcr_quartile",
                "cas_major_tier",
                "cas_top",
                "publication_types",
                "doi",
                "pubmed_url",
            ]
        else:
            show_cols = [
                "pmid",
                "title",
                "journal",
                "pub_year",
                "citation_count",
                "journal_impact_score",
                "jcr_if",
                "jcr_quartile",
                "cas_major_tier",
                "cas_top",
                "publication_types",
                "doi",
                "pubmed_url",
            ]

        editable_cols = ["include_flag", "tags", "note"]
        render_cols = ["include_flag", "tags", "note"] + show_cols
        edited_view = st.data_editor(
            view_df[render_cols],
            use_container_width=True,
            hide_index=True,
            key=f"search_table_editor_{selected_run_id}",
            disabled=[c for c in render_cols if c not in editable_cols],
            column_config={
                "include_flag": st.column_config.CheckboxColumn("Include"),
                "tags": st.column_config.TextColumn("Tags", help="e.g. background;methods;gap"),
                "note": st.column_config.TextColumn("Note"),
            },
        )

        csave1, csave2 = st.columns([1, 3])
        with csave1:
            if st.button("Save inclusion changes", key=f"save_inclusion_{selected_run_id}"):
                repo.save_included_papers(int(selected_run_id), edited_view[["pmid", "include_flag", "tags", "note"]].to_dict(orient="records"))
                st.success("Inclusion set updated from current table view.")
                st.rerun()
        with csave2:
            included_now = int(edited_view["include_flag"].sum()) if "include_flag" in edited_view.columns else 0
            st.caption(f"Included in current view: {included_now}")

        st.markdown("**Abstract viewer**")
        pmid_options = view_df["pmid"].astype(str).tolist()
        selected_pmid = st.selectbox("Select PMID", options=pmid_options)
        selected_row = view_df[view_df["pmid"].astype(str) == selected_pmid].iloc[0]
        st.markdown(f"**Title:** {selected_row['title']}")
        st.markdown(f"**Journal / Year:** {selected_row['journal']} / {selected_row['pub_year']}")
        st.markdown(
            f"**Citation / JCR IF:** {selected_row.get('citation_count', 'N/A')} / "
            f"{selected_row.get('jcr_if', 'N/A')} ({selected_row.get('jcr_quartile', '-')})"
        )
        st.markdown(
            f"**CAS分区:** 大类{selected_row.get('cas_major_tier', 'N/A')} | "
            f"Top={selected_row.get('cas_top', 'N/A')} | 年份={selected_row.get('cas_year', 'N/A')}"
        )
        st.markdown(f"**PMID link:** https://pubmed.ncbi.nlm.nih.gov/{selected_pmid}/")

        abstract_text = selected_row.get("abstract", "") or "No abstract text returned."
        st.text_area("Abstract", value=abstract_text, height=220)

        tcol1, tcol2 = st.columns([1, 3])
        with tcol1:
            if st.button("Translate to Chinese", key=f"translate_abs_{selected_run_id}_{selected_pmid}"):
                if not str(abstract_text).strip() or abstract_text == "No abstract text returned.":
                    st.warning("No abstract text to translate.")
                else:
                    api_base = st.session_state.get("v1_ti_api_base", "")
                    api_key = st.session_state.get("v1_ti_api_key", "")
                    model_name = st.session_state.get("v1_ti_model_name", "")
                    if not api_base or not api_key or not model_name:
                        st.warning("Please configure LLM settings in Topic Ideas first.")
                    else:
                        with st.spinner("Translating abstract..."):
                            try:
                                zh_text = _translate_abstract_to_zh(
                                    abstract_text=str(abstract_text),
                                    api_base=str(api_base),
                                    api_key=str(api_key),
                                    model_name=str(model_name),
                                )
                                st.session_state[f"abs_zh_{selected_run_id}_{selected_pmid}"] = zh_text
                                st.success("Chinese translation generated.")
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Translation failed: {exc}")
        with tcol2:
            if st.button("Clear translation", key=f"clear_abs_zh_{selected_run_id}_{selected_pmid}"):
                st.session_state.pop(f"abs_zh_{selected_run_id}_{selected_pmid}", None)
                st.rerun()

        zh_key = f"abs_zh_{selected_run_id}_{selected_pmid}"
        if zh_key in st.session_state and str(st.session_state.get(zh_key, "")).strip():
            st.text_area("Abstract (Chinese)", value=st.session_state.get(zh_key, ""), height=220)

        st.download_button(
            "Download current papers CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"run_{selected_run_id}_papers.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Evidence Landscape")
        maps = evidence_map(df)

        c1, c2 = st.columns(2)
        with c1:
            bar_chart(maps["disease"], "Top Disease / MeSH Signals")
            bar_chart(maps["intervention"], "Intervention Signals")
        with c2:
            bar_chart(maps["outcome"], "Outcome Signals")
            bar_chart(maps["study_design"], "Study Design Signals")

    with tab3:
        st.subheader("Publication Trend and Emerging Terms")
        trends = compute_trends(df)

        if trends.yearly_counts.empty:
            st.info("Year information is insufficient for trend analysis.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(trends.yearly_counts["year"], trends.yearly_counts["papers"], marker="o", color="#1f5673")
            ax.set_xlabel("Year")
            ax.set_ylabel("Papers")
            ax.set_title("Publication Trend")
            ax.grid(alpha=0.2)
            st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Terms**")
            st.dataframe(trends.top_keywords.head(20), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Emerging Terms (Recent/Past Growth)**")
            st.dataframe(trends.growth_keywords.head(20), hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Candidate Research Topics")
        st.caption("Integrated pipeline: rule engine + optional LLM enhancement -> final topic ideas")

        base_clinical = float(run_info.get("score_weight_clinical", DEFAULT_SCORE_WEIGHTS["clinical_value"]))
        base_innovation = float(run_info.get("score_weight_innovation", DEFAULT_SCORE_WEIGHTS["innovation"]))
        base_feasibility = float(run_info.get("score_weight_feasibility", DEFAULT_SCORE_WEIGHTS["feasibility"]))

        c1, c2, c3 = st.columns(3)
        w_clinical = c1.slider("Clinical value", 0, 100, int(base_clinical * 100), 5, key=f"rw_c_{selected_run_id}")
        w_innovation = c2.slider("Innovation", 0, 100, int(base_innovation * 100), 5, key=f"rw_i_{selected_run_id}")
        w_feasibility = c3.slider("Feasibility", 0, 100, int(base_feasibility * 100), 5, key=f"rw_f_{selected_run_id}")

        weight_sum = w_clinical + w_innovation + w_feasibility
        if weight_sum == 0:
            current_weights = dict(DEFAULT_SCORE_WEIGHTS)
            st.warning("All weights are 0. Fallback to default weights.")
        else:
            current_weights = {
                "clinical_value": w_clinical / weight_sum,
                "innovation": w_innovation / weight_sum,
                "feasibility": w_feasibility / weight_sum,
            }

        use_included_only = st.checkbox("Use only manually included papers if available", value=True, key=f"ti_use_included_{selected_run_id}")
        included_df = repo.get_included_papers(int(selected_run_id))
        base_df = included_df if use_included_only and not included_df.empty else df
        st.caption(f"Evidence source: {'Included set' if (use_included_only and not included_df.empty) else 'Current run papers'} | records={len(base_df)}")

        evidence_export_cols = [c for c in ["pmid", "title", "abstract", "journal", "pub_year", "publication_types", "mesh_terms", "keywords", "doi"] if c in base_df.columns]
        st.download_button(
            "Download topic evidence package (JSONL)",
            data="\n".join(json.dumps(row, ensure_ascii=False) for row in base_df[evidence_export_cols].fillna("").to_dict(orient="records")).encode("utf-8"),
            file_name=f"run_{selected_run_id}_topic_evidence.jsonl",
            mime="application/json",
        )

        enable_llm = st.checkbox("Enable LLM enhancement", value=False, key=f"ti_enable_llm_{selected_run_id}")
        llm_error = ""
        api_base = ""
        api_key = ""
        model_name = ""
        temperature = 0.1

        if enable_llm:
            with st.expander("LLM settings", expanded=True):
                profiles_df = repo.list_llm_profiles()
                if not profiles_df.empty:
                    st.caption("Saved interfaces")
                    st.dataframe(
                        profiles_df[["profile_name", "api_base", "model_name", "is_default", "updated_at"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                profile_name_input = st.text_input(
                    "接口名称",
                    value=st.session_state.get("v1_ti_profile_name", "default"),
                    key=f"ti_profile_name_{selected_run_id}",
                )
                api_base = st.text_input(
                    "Base URL",
                    value=st.session_state.get("v1_ti_api_base", "https://api.openai.com/v1"),
                    key=f"ti_api_base_{selected_run_id}",
                )
                model_name = st.text_input(
                    "Model",
                    value=st.session_state.get("v1_ti_model_name", "gpt-4o-mini"),
                    key=f"ti_model_name_{selected_run_id}",
                )
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=st.session_state.get("v1_ti_api_key", ""),
                    key=f"ti_api_key_{selected_run_id}",
                )
                temperature = st.slider("LLM temperature", min_value=0.0, max_value=0.8, value=0.1, step=0.05, key=f"ti_temp_{selected_run_id}")
                set_default = st.checkbox("Set as default profile", value=bool(st.session_state.get("v1_ti_profile_id", 0)), key=f"ti_profile_default_{selected_run_id}")

                save_col1, save_col2, save_col3 = st.columns(3)
                with save_col1:
                    if st.button("Load by 接口名称", key=f"ti_profile_load_{selected_run_id}"):
                        if profiles_df.empty:
                            st.warning("No saved interfaces.")
                        else:
                            matched = profiles_df[profiles_df["profile_name"] == profile_name_input.strip()]
                            if matched.empty:
                                st.warning("No matching 接口名称.")
                            else:
                                row = matched.iloc[0]
                                st.session_state["v1_ti_profile_id"] = int(row.get("id", 0) or 0)
                                st.session_state["v1_ti_profile_name"] = str(row.get("profile_name", ""))
                                st.session_state["v1_ti_api_base"] = str(row.get("api_base", ""))
                                st.session_state["v1_ti_model_name"] = str(row.get("model_name", ""))
                                st.session_state["v1_ti_api_key"] = str(row.get("api_key", ""))
                                st.success("Loaded interface config.")
                                st.rerun()
                with save_col2:
                    if st.button("Save new profile", key=f"ti_profile_save_new_{selected_run_id}"):
                        new_id = repo.upsert_llm_profile(
                            provider_name="OpenAI-compatible",
                            profile_name=profile_name_input.strip() or "default",
                            api_base=api_base.strip(),
                            api_key=api_key.strip(),
                            model_name=model_name.strip(),
                            profile_id=None,
                            is_default=set_default,
                        )
                        st.session_state["v1_ti_profile_id"] = new_id
                        st.session_state["v1_ti_profile_name"] = profile_name_input.strip() or "default"
                        st.success("Saved profile.")
                        st.rerun()
                with save_col3:
                    if st.button("Update selected profile", key=f"ti_profile_update_{selected_run_id}"):
                        profile_id = int(st.session_state.get("v1_ti_profile_id", 0) or 0)
                        if profile_id <= 0:
                            st.warning("No saved profile selected to update.")
                        else:
                            repo.upsert_llm_profile(
                                provider_name="OpenAI-compatible",
                                profile_name=profile_name_input.strip() or "default",
                                api_base=api_base.strip(),
                                api_key=api_key.strip(),
                                model_name=model_name.strip(),
                                profile_id=profile_id,
                                is_default=set_default,
                            )
                            st.success("Updated profile.")
                            st.rerun()

                st.session_state["v1_ti_profile_name"] = profile_name_input.strip() or "default"
                st.session_state["v1_ti_api_base"] = api_base
                st.session_state["v1_ti_model_name"] = model_name
                st.session_state["v1_ti_api_key"] = api_key
                st.caption("Medical prompt enforces evidence-grounded output with PMID support.")

        if st.button("Generate final topic ideas", type="primary", key=f"ti_generate_{selected_run_id}"):
            rule_topics = generate_topic_candidates(
                base_df,
                top_n=5,
                score_weights=current_weights,
                query_text=str(run_info.get("raw_query", "")) or str(run_info.get("expanded_query", "")),
            )

            final_topics = rule_topics
            if enable_llm:
                try:
                    evidence_records = base_df[evidence_export_cols].fillna("").to_dict(orient="records")
                    llm_topics = enhance_topic_ideas_with_llm(
                        api_base=api_base,
                        api_key=api_key,
                        model_name=model_name,
                        query_text=str(run_info.get("raw_query", "")) or str(run_info.get("expanded_query", "")),
                        rule_topics=rule_topics,
                        evidence_records=evidence_records,
                        top_n=5,
                        temperature=float(temperature),
                    )

                    score_lookup = {t.get("topic_title", ""): t for t in rule_topics}
                    merged_topics = []
                    for idx, item in enumerate(llm_topics):
                        base_match = score_lookup.get(item.get("topic_title", ""), {})
                        merged_topics.append(
                            {
                                "topic_title": item.get("topic_title", ""),
                                "rationale": item.get("rationale", ""),
                                "key_gap": item.get("key_gap", ""),
                                "score": float(item.get("score", base_match.get("score", max(0, 95 - idx * 8)))),
                                "component_clinical_value": float(base_match.get("component_clinical_value", 0)),
                                "component_innovation": float(base_match.get("component_innovation", 0)),
                                "component_feasibility": float(base_match.get("component_feasibility", 0)),
                                "supporting_pmids": item.get("supporting_pmids", []) or base_match.get("supporting_pmids", []),
                            }
                        )
                    final_topics = merged_topics if merged_topics else rule_topics
                except Exception as exc:  # noqa: BLE001
                    llm_error = str(exc)
                    final_topics = rule_topics

            repo.save_topic_candidates(int(selected_run_id), final_topics)
            st.session_state[f"ti_generation_msg_{selected_run_id}"] = "Generated and saved final topic ideas."
            st.session_state[f"ti_generation_err_{selected_run_id}"] = llm_error
            st.rerun()

        msg = st.session_state.get(f"ti_generation_msg_{selected_run_id}", "")
        err = st.session_state.get(f"ti_generation_err_{selected_run_id}", "")
        if msg:
            st.success(msg)
        if err:
            st.warning(f"LLM enhancement fallback to rule engine: {err}")

        topic_df = repo.get_topic_candidates(int(selected_run_id))
        if topic_df.empty:
            st.info("No topic candidates yet. Click 'Generate final topic ideas'.")
        else:
            base_records_for_chat = base_df[evidence_export_cols].fillna("").to_dict(orient="records")
            full_run_records = df[evidence_export_cols].fillna("").to_dict(orient="records")
            pmid_record_map = {
                _normalize_pmid(rec.get("pmid", "")): rec
                for rec in base_records_for_chat
                if _normalize_pmid(rec.get("pmid", ""))
            }
            full_run_pmid_map = {
                _normalize_pmid(rec.get("pmid", "")): rec
                for rec in full_run_records
                if _normalize_pmid(rec.get("pmid", ""))
            }
            for idx, row in topic_df.iterrows():
                topic_title = str(row.get("topic_title", ""))
                supporting_pmids = [_normalize_pmid(p) for p in str(row.get("supporting_pmids", "")).split(";") if _normalize_pmid(p)]
                topic_evidence = [pmid_record_map[p] for p in supporting_pmids if p in pmid_record_map]
                if not topic_evidence:
                    topic_evidence = base_records_for_chat[:20]

                with st.container(border=True):
                    st.markdown(f"### {idx + 1}. {topic_title}")
                    st.markdown(f"**Priority score:** {row.get('score', '-')}")
                    st.markdown(f"**Why this topic:** {row.get('rationale', '')}")
                    st.markdown(f"**Potential gap:** {row.get('key_gap', '')}")
                    st.markdown(f"**Supporting PMIDs:** {row.get('supporting_pmids', '')}")

                    with st.expander("Continue discussion with LLM", expanded=False):
                        chat_df = repo.get_topic_followups(int(selected_run_id), topic_title)
                        if chat_df.empty:
                            st.caption("No discussion yet. Ask your first follow-up question.")
                            history_pairs: list[tuple[str, str]] = []
                        else:
                            history_pairs = []
                            for _, hrow in chat_df.iterrows():
                                uq = str(hrow.get("user_message", ""))
                                aa = str(hrow.get("assistant_message", ""))
                                history_pairs.append((uq, aa))
                                st.markdown(f"**You:** {uq}")
                                st.markdown(f"**LLM:** {aa}")
                                st.markdown("---")

                        followup_input_key = f"topic_followup_q_{selected_run_id}_{idx}"
                        user_q = st.text_area(
                            "Your follow-up question",
                            value="",
                            height=90,
                            key=followup_input_key,
                            placeholder="e.g. How to narrow this topic for a retrospective cohort study?",
                        )

                        send_col1, send_col2 = st.columns([1, 3])
                        with send_col1:
                            if st.button("Send to LLM", key=f"send_topic_followup_{selected_run_id}_{idx}"):
                                api_base_chat = str(st.session_state.get("v1_ti_api_base", "")).strip()
                                api_key_chat = str(st.session_state.get("v1_ti_api_key", "")).strip()
                                model_name_chat = str(st.session_state.get("v1_ti_model_name", "")).strip()

                                if not api_base_chat or not api_key_chat or not model_name_chat:
                                    st.warning("Please set and save LLM interface config first in LLM settings.")
                                elif not user_q.strip():
                                    st.warning("Please enter a question.")
                                else:
                                    requested_pmids = _extract_pmids_from_text(user_q)
                                    requested_records = [full_run_pmid_map[p] for p in requested_pmids if p in full_run_pmid_map]
                                    missing_pmids = [p for p in requested_pmids if p not in full_run_pmid_map]
                                    if missing_pmids:
                                        st.info("These PMIDs are not in this run dataset: " + ", ".join(missing_pmids))

                                    evidence_for_prompt: list[dict] = []
                                    seen_pmids = set()
                                    for rec in requested_records + topic_evidence:
                                        key = _normalize_pmid(rec.get("pmid", ""))
                                        if not key or key in seen_pmids:
                                            continue
                                        seen_pmids.add(key)
                                        evidence_for_prompt.append(rec)

                                    prompt = _build_topic_followup_prompt(
                                        topic_title=topic_title,
                                        rationale=str(row.get("rationale", "")),
                                        key_gap=str(row.get("key_gap", "")),
                                        supporting_pmids=supporting_pmids,
                                        evidence_records=evidence_for_prompt,
                                        requested_records=requested_records,
                                        history_pairs=history_pairs,
                                        user_message=user_q.strip(),
                                    )
                                    try:
                                        with st.spinner("LLM is generating answer..."):
                                            answer = call_openai_compatible(
                                                api_base=api_base_chat,
                                                api_key=api_key_chat,
                                                model_name=model_name_chat,
                                                prompt=prompt,
                                                temperature=0.2,
                                                timeout=180,
                                            )
                                        repo.save_topic_followup(
                                            run_id=int(selected_run_id),
                                            topic_title=topic_title,
                                            user_message=user_q.strip(),
                                            assistant_message=answer,
                                        )
                                        st.success("Discussion saved.")
                                        st.rerun()
                                    except Exception as exc:  # noqa: BLE001
                                        st.error(f"Follow-up failed: {exc}")

                        final_key = f"topic_final_opinion_{selected_run_id}_{idx}"
                        if final_key not in st.session_state:
                            st.session_state[final_key] = repo.get_topic_final_opinion(int(selected_run_id), topic_title)

                        st.text_area(
                            "Final agreed opinion (saved locally)",
                            key=final_key,
                            height=120,
                            placeholder="Write the final decision after discussion, e.g. selected endpoint, cohort design, inclusion criteria focus.",
                        )
                        if st.button("Save final opinion", key=f"save_topic_final_{selected_run_id}_{idx}"):
                            repo.upsert_topic_final_opinion(
                                run_id=int(selected_run_id),
                                topic_title=topic_title,
                                final_opinion=str(st.session_state.get(final_key, "")).strip(),
                            )
                            st.success("Final opinion saved to local database.")


if __name__ == "__main__":
    main()
