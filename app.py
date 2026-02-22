from __future__ import annotations

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
    c2.metric("Unique Journals", df["journal"].nunique())

    year_series = pd.to_numeric(df["pub_year"], errors="coerce").dropna()
    c3.metric("Median Year", int(year_series.median()) if not year_series.empty else "-")

    c4.metric("With DOI", int(df["doi"].fillna("").astype(str).str.len().gt(0).sum()))


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

    st.session_state["last_query"] = query

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

        try:
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
        except TypeError:
            # Backward compatibility for stale Streamlit worker using older Repository signature.
            run_id = repo.create_run(
                project_name=project_name.strip(),
                raw_query=keywords.strip(),
                expanded_query=query,
                start_year=int(start_year),
                end_year=int(end_year),
                article_types=selected_types,
                max_results=int(max_results),
                total_results=total_found,
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


def main() -> None:
    st.set_page_config(page_title="Clinical Topic Discovery", page_icon="🩺", layout="wide")
    st.title("Clinical Topic Discovery")
    st.caption("V1: PubMed-based topic discovery for clinical paper planning")

    repo = get_repo()
    default_search_weights = dict(DEFAULT_SCORE_WEIGHTS)

    new_run_id = run_search(repo, score_weights=default_search_weights)

    runs = repo.get_runs()
    if runs.empty:
        st.info("No run yet. Use the sidebar to start your first PubMed analysis.")
        return

    runs = runs.copy()
    runs["created_at_local"] = runs["created_at"].astype(str).apply(_format_created_at_local)

    with st.expander("History Management", expanded=False):
        st.caption("Use these actions to keep the run list short and clean.")
        confirm_clear = st.checkbox("I understand this will permanently delete all run history.", key="confirm_clear_all")
        if st.button("Clear all history", key="clear_all_history_btn"):
            if confirm_clear:
                repo.clear_history()
                st.success("All history cleared.")
                st.rerun()
            else:
                st.warning("Please confirm before clearing all history.")

    filter_text = st.text_input("Filter runs (project/query)", value="", placeholder="Type to filter history list")
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

    max_default = min(50, len(run_view))
    show_limit = st.number_input("Show recent runs", min_value=1, max_value=max(1, len(run_view)), value=max_default, step=1)
    run_view = run_view.head(int(show_limit))

    run_options = run_view["id"].tolist()
    default_idx = 0
    if new_run_id and new_run_id in run_options:
        default_idx = run_options.index(new_run_id)

    selected_run_id = st.selectbox(
        "Select analysis run",
        options=run_options,
        index=default_idx,
        format_func=lambda rid: (
            f"Run #{rid} | "
            f"{run_view.loc[run_view['id'] == rid, 'project_name'].iloc[0]} | "
            f"{run_view.loc[run_view['id'] == rid, 'created_at_local'].iloc[0]}"
        ),
    )

    col_delete, col_confirm = st.columns([1, 3])
    with col_confirm:
        confirm_delete = st.checkbox("Confirm delete selected run", key=f"confirm_delete_run_{selected_run_id}")
    with col_delete:
        if st.button("Delete selected run", key=f"delete_run_btn_{selected_run_id}"):
            if confirm_delete:
                repo.delete_run(int(selected_run_id))
                st.success(f"Run #{selected_run_id} deleted.")
                st.rerun()
            else:
                st.warning("Please confirm deletion first.")

    run_info = runs[runs["id"] == selected_run_id].iloc[0]
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
            st.caption(
                "Citation count is from OpenAlex. JCR IF and CAS partition are integrated from ShowJCR's local jcr.db tables."
            )

        years = sorted(pd.to_numeric(df["pub_year"], errors="coerce").dropna().astype(int).unique().tolist())
        selected_years = st.multiselect("Filter by year", options=years, default=years[-5:] if len(years) > 5 else years)

        filtered = df.copy()
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

        st.dataframe(view_df[show_cols], use_container_width=True, hide_index=True)

        st.markdown("**Abstract viewer**")
        pmid_options = filtered["pmid"].astype(str).tolist()
        selected_pmid = st.selectbox("Select PMID", options=pmid_options)
        selected_row = filtered[filtered["pmid"].astype(str) == selected_pmid].iloc[0]
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
        st.text_area("Abstract", value=selected_row.get("abstract", "") or "No abstract text returned.", height=220)

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

        base_clinical = float(run_info.get("score_weight_clinical", DEFAULT_SCORE_WEIGHTS["clinical_value"]))
        base_innovation = float(run_info.get("score_weight_innovation", DEFAULT_SCORE_WEIGHTS["innovation"]))
        base_feasibility = float(run_info.get("score_weight_feasibility", DEFAULT_SCORE_WEIGHTS["feasibility"]))

        st.markdown("**Re-score this run with custom weights**")
        st.caption("These weights apply to the currently selected run only. They do not change search behavior.")
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

        st.caption(
            f"Normalized: clinical={current_weights['clinical_value']:.2f}, "
            f"innovation={current_weights['innovation']:.2f}, "
            f"feasibility={current_weights['feasibility']:.2f}"
        )

        live_topics = generate_topic_candidates(
            df,
            top_n=5,
            score_weights=current_weights,
            query_text=str(run_info.get("raw_query", "")) or str(run_info.get("expanded_query", "")),
        )

        if st.button("Save current ranking to this run", key=f"save_rescore_btn_{selected_run_id}"):
            repo.save_topic_candidates(int(selected_run_id), live_topics)
            st.success("Saved current ranking.")

        if not live_topics:
            st.info("No topic candidates generated. Try broader filters or a larger sample.")
        else:
            score_table = pd.DataFrame(
                [
                    {
                        "rank": idx + 1,
                        "topic": t["topic_title"],
                        "score": t["score"],
                        "clinical": t["component_clinical_value"],
                        "innovation": t["component_innovation"],
                        "feasibility": t["component_feasibility"],
                    }
                    for idx, t in enumerate(live_topics)
                ]
            )
            st.dataframe(score_table, use_container_width=True, hide_index=True)

            for idx, topic in enumerate(live_topics):
                with st.container(border=True):
                    st.markdown(f"### {idx + 1}. {topic['topic_title']}")
                    st.markdown(f"**Priority score:** {topic['score']}")
                    st.markdown(
                        "**Component scores:** "
                        f"Clinical={topic.get('component_clinical_value', '-')}, "
                        f"Innovation={topic.get('component_innovation', '-')}, "
                        f"Feasibility={topic.get('component_feasibility', '-')}"
                    )
                    st.markdown(f"**Why this topic:** {topic['rationale']}")
                    st.markdown(f"**Potential gap:** {topic['key_gap']}")
                    st.markdown(f"**Supporting PMIDs:** {'; '.join(topic['supporting_pmids'])}")


if __name__ == "__main__":
    main()
