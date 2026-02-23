from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


class Repository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column_def: str) -> None:
        column_name = column_def.split()[0]
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    project_name TEXT,
                    raw_query TEXT,
                    expanded_query TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    article_types TEXT,
                    max_results INTEGER,
                    total_results INTEGER,
                    score_weight_clinical REAL DEFAULT 0.4,
                    score_weight_innovation REAL DEFAULT 0.35,
                    score_weight_feasibility REAL DEFAULT 0.25
                );

                CREATE TABLE IF NOT EXISTS papers (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    journal TEXT,
                    pub_date TEXT,
                    pub_year TEXT,
                    doi TEXT,
                    authors TEXT,
                    publication_types TEXT,
                    mesh_terms TEXT,
                    keywords TEXT,
                    citation_count INTEGER,
                    journal_impact_score REAL,
                    impact_source TEXT,
                    jcr_if REAL,
                    jcr_quartile TEXT,
                    jcr_year TEXT,
                    cas_major_category TEXT,
                    cas_major_tier INTEGER,
                    cas_small_category TEXT,
                    cas_small_tier INTEGER,
                    cas_top INTEGER,
                    cas_year TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS run_papers (
                    run_id INTEGER,
                    pmid TEXT,
                    PRIMARY KEY (run_id, pmid),
                    FOREIGN KEY (run_id) REFERENCES search_runs(id),
                    FOREIGN KEY (pmid) REFERENCES papers(pmid)
                );

                CREATE TABLE IF NOT EXISTS topic_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    topic_title TEXT,
                    rationale TEXT,
                    key_gap TEXT,
                    score REAL,
                    component_clinical_value REAL,
                    component_innovation REAL,
                    component_feasibility REAL,
                    supporting_pmids TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES search_runs(id)
                );

                CREATE TABLE IF NOT EXISTS included_papers (
                    run_id INTEGER,
                    pmid TEXT,
                    include_flag INTEGER DEFAULT 1,
                    tags TEXT,
                    note TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, pmid),
                    FOREIGN KEY (run_id) REFERENCES search_runs(id),
                    FOREIGN KEY (pmid) REFERENCES papers(pmid)
                );

                CREATE TABLE IF NOT EXISTS llm_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_name TEXT,
                    profile_name TEXT UNIQUE,
                    api_base TEXT,
                    api_key TEXT,
                    model_name TEXT,
                    is_default INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS topic_followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    topic_title TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES search_runs(id)
                );

                CREATE TABLE IF NOT EXISTS topic_final_opinions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    topic_title TEXT,
                    final_opinion TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (run_id, topic_title),
                    FOREIGN KEY (run_id) REFERENCES search_runs(id)
                );

                """
            )

            self._ensure_column(conn, "search_runs", "score_weight_clinical REAL DEFAULT 0.4")
            self._ensure_column(conn, "search_runs", "score_weight_innovation REAL DEFAULT 0.35")
            self._ensure_column(conn, "search_runs", "score_weight_feasibility REAL DEFAULT 0.25")
            self._ensure_column(conn, "topic_candidates", "component_clinical_value REAL")
            self._ensure_column(conn, "topic_candidates", "component_innovation REAL")
            self._ensure_column(conn, "topic_candidates", "component_feasibility REAL")
            self._ensure_column(conn, "papers", "citation_count INTEGER")
            self._ensure_column(conn, "papers", "journal_impact_score REAL")
            self._ensure_column(conn, "papers", "impact_source TEXT")
            self._ensure_column(conn, "papers", "jcr_if REAL")
            self._ensure_column(conn, "papers", "jcr_quartile TEXT")
            self._ensure_column(conn, "papers", "jcr_year TEXT")
            self._ensure_column(conn, "papers", "cas_major_category TEXT")
            self._ensure_column(conn, "papers", "cas_major_tier INTEGER")
            self._ensure_column(conn, "papers", "cas_small_category TEXT")
            self._ensure_column(conn, "papers", "cas_small_tier INTEGER")
            self._ensure_column(conn, "papers", "cas_top INTEGER")
            self._ensure_column(conn, "papers", "cas_year TEXT")

    def create_run(
        self,
        project_name: str,
        raw_query: str,
        expanded_query: str,
        start_year: int,
        end_year: int,
        article_types: Iterable[str],
        max_results: int,
        total_results: int,
        score_weights: dict[str, float] | None = None,
    ) -> int:
        score_weights = score_weights or {}
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO search_runs
                (project_name, raw_query, expanded_query, start_year, end_year, article_types, max_results, total_results,
                 score_weight_clinical, score_weight_innovation, score_weight_feasibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_name,
                    raw_query,
                    expanded_query,
                    start_year,
                    end_year,
                    "; ".join(article_types),
                    max_results,
                    total_results,
                    score_weights.get("clinical_value", 0.4),
                    score_weights.get("innovation", 0.35),
                    score_weights.get("feasibility", 0.25),
                ),
            )
            return int(cur.lastrowid)

    def upsert_papers(self, records: list[dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO papers
                (pmid, title, abstract, journal, pub_date, pub_year, doi, authors, publication_types, mesh_terms, keywords,
                 citation_count, journal_impact_score, impact_source, jcr_if, jcr_quartile, jcr_year,
                 cas_major_category, cas_major_tier, cas_small_category, cas_small_tier, cas_top, cas_year)
                VALUES (:pmid, :title, :abstract, :journal, :pub_date, :pub_year, :doi, :authors, :publication_types, :mesh_terms, :keywords,
                        COALESCE(:citation_count, NULL), COALESCE(:journal_impact_score, NULL), COALESCE(:impact_source, NULL),
                        COALESCE(:jcr_if, NULL), COALESCE(:jcr_quartile, NULL), COALESCE(:jcr_year, NULL),
                        COALESCE(:cas_major_category, NULL), COALESCE(:cas_major_tier, NULL), COALESCE(:cas_small_category, NULL),
                        COALESCE(:cas_small_tier, NULL), COALESCE(:cas_top, NULL), COALESCE(:cas_year, NULL))
                ON CONFLICT(pmid) DO UPDATE SET
                    title = excluded.title,
                    abstract = excluded.abstract,
                    journal = excluded.journal,
                    pub_date = excluded.pub_date,
                    pub_year = excluded.pub_year,
                    doi = excluded.doi,
                    authors = excluded.authors,
                    publication_types = excluded.publication_types,
                    mesh_terms = excluded.mesh_terms,
                    keywords = excluded.keywords,
                    citation_count = COALESCE(excluded.citation_count, papers.citation_count),
                    journal_impact_score = COALESCE(excluded.journal_impact_score, papers.journal_impact_score),
                    impact_source = COALESCE(excluded.impact_source, papers.impact_source),
                    jcr_if = COALESCE(excluded.jcr_if, papers.jcr_if),
                    jcr_quartile = COALESCE(excluded.jcr_quartile, papers.jcr_quartile),
                    jcr_year = COALESCE(excluded.jcr_year, papers.jcr_year),
                    cas_major_category = COALESCE(excluded.cas_major_category, papers.cas_major_category),
                    cas_major_tier = COALESCE(excluded.cas_major_tier, papers.cas_major_tier),
                    cas_small_category = COALESCE(excluded.cas_small_category, papers.cas_small_category),
                    cas_small_tier = COALESCE(excluded.cas_small_tier, papers.cas_small_tier),
                    cas_top = COALESCE(excluded.cas_top, papers.cas_top),
                    cas_year = COALESCE(excluded.cas_year, papers.cas_year),
                    updated_at = CURRENT_TIMESTAMP
                """,
                records,
            )

    def link_run_papers(self, run_id: int, pmids: list[str]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO run_papers (run_id, pmid) VALUES (?, ?)",
                [(run_id, pmid) for pmid in pmids],
            )

    def update_paper_metrics(self, metrics: list[dict]) -> None:
        if not metrics:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE papers
                SET citation_count = :citation_count,
                    journal_impact_score = :journal_impact_score,
                    impact_source = :impact_source,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pmid = :pmid
                """,
                metrics,
            )

    def update_paper_jcr(self, metrics: list[dict]) -> None:
        if not metrics:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE papers
                SET jcr_if = :jcr_if,
                    jcr_quartile = :jcr_quartile,
                    jcr_year = :jcr_year,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pmid = :pmid
                """,
                metrics,
            )

    def update_paper_cas(self, metrics: list[dict]) -> None:
        if not metrics:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE papers
                SET cas_major_category = :cas_major_category,
                    cas_major_tier = :cas_major_tier,
                    cas_small_category = :cas_small_category,
                    cas_small_tier = :cas_small_tier,
                    cas_top = :cas_top,
                    cas_year = :cas_year,
                    updated_at = CURRENT_TIMESTAMP
                WHERE pmid = :pmid
                """,
                metrics,
            )

    def save_topic_candidates(self, run_id: int, topics: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM topic_candidates WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO topic_candidates
                (run_id, topic_title, rationale, key_gap, score, component_clinical_value, component_innovation, component_feasibility, supporting_pmids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        topic["topic_title"],
                        topic["rationale"],
                        topic["key_gap"],
                        topic["score"],
                        topic.get("component_clinical_value"),
                        topic.get("component_innovation"),
                        topic.get("component_feasibility"),
                        "; ".join(topic["supporting_pmids"]),
                    )
                    for topic in topics
                ],
            )

    def get_runs(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    id,
                    created_at,
                    project_name,
                    raw_query,
                    expanded_query,
                    total_results,
                    score_weight_clinical,
                    score_weight_innovation,
                    score_weight_feasibility
                FROM search_runs
                ORDER BY id DESC
                """,
                conn,
            )

    def get_run_papers(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT p.*
                FROM papers p
                JOIN run_papers rp ON p.pmid = rp.pmid
                WHERE rp.run_id = ?
                ORDER BY CAST(COALESCE(NULLIF(p.pub_year, ''), '0') AS INTEGER) DESC, p.pmid DESC
                """,
                conn,
                params=(run_id,),
            )

    def get_topic_candidates(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    topic_title,
                    rationale,
                    key_gap,
                    score,
                    component_clinical_value,
                    component_innovation,
                    component_feasibility,
                    supporting_pmids
                FROM topic_candidates
                WHERE run_id = ?
                ORDER BY score DESC
                """,
                conn,
                params=(run_id,),
            )

    def save_included_papers(self, run_id: int, rows: list[dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO included_papers (run_id, pmid, include_flag, tags, note, updated_at)
                VALUES (:run_id, :pmid, :include_flag, :tags, :note, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id, pmid) DO UPDATE SET
                    include_flag = excluded.include_flag,
                    tags = excluded.tags,
                    note = excluded.note,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    {
                        "run_id": run_id,
                        "pmid": str(item.get("pmid", "")),
                        "include_flag": 1 if bool(item.get("include_flag", False)) else 0,
                        "tags": str(item.get("tags", "") or ""),
                        "note": str(item.get("note", "") or ""),
                    }
                    for item in rows
                    if str(item.get("pmid", "")).strip()
                ],
            )

    def get_included_papers(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    p.*,
                    ip.include_flag,
                    ip.tags,
                    ip.note,
                    ip.updated_at AS include_updated_at
                FROM included_papers ip
                JOIN papers p ON p.pmid = ip.pmid
                WHERE ip.run_id = ? AND ip.include_flag = 1
                ORDER BY CAST(COALESCE(NULLIF(p.pub_year, ''), '0') AS INTEGER) DESC, p.pmid DESC
                """,
                conn,
                params=(run_id,),
            )

    def get_inclusion_table(self, run_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT
                    p.pmid,
                    p.title,
                    p.journal,
                    p.pub_year,
                    COALESCE(ip.include_flag, 0) AS include_flag,
                    COALESCE(ip.tags, '') AS tags,
                    COALESCE(ip.note, '') AS note
                FROM run_papers rp
                JOIN papers p ON p.pmid = rp.pmid
                LEFT JOIN included_papers ip ON ip.run_id = rp.run_id AND ip.pmid = rp.pmid
                WHERE rp.run_id = ?
                ORDER BY CAST(COALESCE(NULLIF(p.pub_year, ''), '0') AS INTEGER) DESC, p.pmid DESC
                """,
                conn,
                params=(run_id,),
            )

    def list_llm_profiles(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT id, provider_name, profile_name, api_base, api_key, model_name, is_default, updated_at
                FROM llm_profiles
                ORDER BY is_default DESC, updated_at DESC, id DESC
                """,
                conn,
            )

    def get_default_llm_profile(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, provider_name, profile_name, api_base, api_key, model_name, is_default, updated_at
                FROM llm_profiles
                WHERE is_default = 1
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "provider_name": row[1],
            "profile_name": row[2],
            "api_base": row[3],
            "api_key": row[4],
            "model_name": row[5],
            "is_default": row[6],
            "updated_at": row[7],
        }

    def upsert_llm_profile(
        self,
        provider_name: str,
        profile_name: str,
        api_base: str,
        api_key: str,
        model_name: str,
        profile_id: int | None = None,
        is_default: bool = False,
    ) -> int:
        with self._connect() as conn:
            if is_default:
                conn.execute("UPDATE llm_profiles SET is_default = 0")

            if profile_id is not None:
                conn.execute(
                    """
                    UPDATE llm_profiles
                    SET provider_name = ?,
                        profile_name = ?,
                        api_base = ?,
                        api_key = ?,
                        model_name = ?,
                        is_default = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        provider_name,
                        profile_name,
                        api_base,
                        api_key,
                        model_name,
                        1 if is_default else 0,
                        profile_id,
                    ),
                )
                return int(profile_id if profile_id is not None else 0)

            existing = conn.execute("SELECT id FROM llm_profiles WHERE profile_name = ?", (profile_name,)).fetchone()
            if existing:
                matched_id = int(existing[0])
                conn.execute(
                    """
                    UPDATE llm_profiles
                    SET provider_name = ?,
                        api_base = ?,
                        api_key = ?,
                        model_name = ?,
                        is_default = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        provider_name,
                        api_base,
                        api_key,
                        model_name,
                        1 if is_default else 0,
                        matched_id,
                    ),
                )
                return matched_id

            cur = conn.execute(
                """
                INSERT INTO llm_profiles (provider_name, profile_name, api_base, api_key, model_name, is_default)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_name,
                    profile_name,
                    api_base,
                    api_key,
                    model_name,
                    1 if is_default else 0,
                ),
            )
            return int(cur.lastrowid)

    def save_topic_followup(
        self,
        run_id: int,
        topic_title: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_followups (run_id, topic_title, user_message, assistant_message)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, topic_title, user_message, assistant_message),
            )

    def get_topic_followups(self, run_id: int, topic_title: str) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT user_message, assistant_message, created_at
                FROM topic_followups
                WHERE run_id = ? AND topic_title = ?
                ORDER BY id ASC
                """,
                conn,
                params=[run_id, topic_title],
            )

    def upsert_topic_final_opinion(self, run_id: int, topic_title: str, final_opinion: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO topic_final_opinions (run_id, topic_title, final_opinion, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(run_id, topic_title) DO UPDATE SET
                    final_opinion = excluded.final_opinion,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (run_id, topic_title, final_opinion),
            )

    def get_topic_final_opinion(self, run_id: int, topic_title: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT final_opinion
                FROM topic_final_opinions
                WHERE run_id = ? AND topic_title = ?
                LIMIT 1
                """,
                (run_id, topic_title),
            ).fetchone()
        return str(row[0]) if row and row[0] is not None else ""

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM topic_candidates WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM included_papers WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM topic_followups WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM topic_final_opinions WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM run_papers WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM search_runs WHERE id = ?", (run_id,))
            conn.execute(
                """
                DELETE FROM papers
                WHERE pmid NOT IN (SELECT DISTINCT pmid FROM run_papers)
                """
            )

    def clear_history(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM topic_candidates")
            conn.execute("DELETE FROM included_papers")
            conn.execute("DELETE FROM topic_followups")
            conn.execute("DELETE FROM topic_final_opinions")
            conn.execute("DELETE FROM run_papers")
            conn.execute("DELETE FROM search_runs")
            conn.execute("DELETE FROM papers")
