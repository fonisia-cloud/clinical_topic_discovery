from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests


SHOWJCR_DB_URL = (
    "https://raw.githubusercontent.com/hitfyd/ShowJCR/master/"
    "%E4%B8%AD%E7%A7%91%E9%99%A2%E5%88%86%E5%8C%BA%E8%A1%A8%E5%8F%8AJCR%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6/jcr.db"
)


def _normalize_journal_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    upper = name.upper().strip()
    chars = []
    for ch in upper:
        if ch.isalnum():
            chars.append(ch)
    return "".join(chars)


def ensure_showjcr_db(db_path: str, force_download: bool = False) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_download:
        return path

    resp = requests.get(SHOWJCR_DB_URL, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _best_quartile(values: Iterable[str]) -> str | None:
    order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    cleaned = []
    for value in values:
        text = str(value).strip().upper()
        if text in order:
            cleaned.append(text)
    if not cleaned:
        return None
    return sorted(cleaned, key=lambda x: order[x])[0]


def _extract_tier(value: object) -> int | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # Handles forms like "1", "Q1", "1 [23/456]"
    for token in text.replace("Q", "").replace("q", "").split():
        if token.isdigit():
            return int(token)

    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if digits:
        return int(digits)
    return None


def load_jcr_index(db_path: str, table_name: str = "JCR2024") -> dict[str, dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        rows = con.execute(
            f'SELECT "Journal", "IF(2024)", "IF Quartile(2024)" FROM {table_name}'
        ).fetchall()
    finally:
        con.close()

    grouped: Dict[str, list[Tuple[float | None, str | None]]] = {}
    for row in rows:
        journal = row[0]
        key = _normalize_journal_name(journal)
        if not key:
            continue

        raw_if = row[1]
        if_value: float | None = None
        try:
            if raw_if is not None and str(raw_if).strip() != "":
                if_value = float(raw_if)
        except Exception:  # noqa: BLE001
            if_value = None

        quartile = str(row[2]).strip() if row[2] is not None else None
        grouped.setdefault(key, []).append((if_value, quartile))

    index: dict[str, dict] = {}
    for key, values in grouped.items():
        ifs = [x[0] for x in values if x[0] is not None]
        quartiles = [x[1] for x in values if x[1]]
        index[key] = {
            "jcr_if": max(ifs) if ifs else None,
            "jcr_quartile": _best_quartile(quartiles),
            "jcr_year": table_name.replace("JCR", ""),
        }

    return index


def load_cas_index(db_path: str, table_name: str = "FQBJCR2025") -> dict[str, dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    try:
        rows = con.execute(f"SELECT * FROM {table_name}").fetchall()
    finally:
        con.close()

    index: dict[str, dict] = {}

    for row in rows:
        values = list(row)
        if not values:
            continue

        journal = values[0]
        key = _normalize_journal_name(journal)
        if not key:
            continue

        # Compatible with observed schemas:
        # FQBJCR2025: major at [8], major tier at [9], top at [10], small1 at [11], small1 tier at [12]
        # FQBJCR2023/2022: major at [6], major tier at [7], top at [8], small1 at [9], small1 tier at [10]
        if len(values) >= 23:
            major_category = values[8]
            major_tier_raw = values[9]
            top_flag_raw = values[10]
            small_category = values[11]
            small_tier_raw = values[12]
        elif len(values) >= 11:
            major_category = values[6]
            major_tier_raw = values[7]
            top_flag_raw = values[8]
            small_category = values[9]
            small_tier_raw = values[10]
        else:
            continue

        major_tier = _extract_tier(major_tier_raw)
        small_tier = _extract_tier(small_tier_raw)
        top_flag = str(top_flag_raw).strip() if top_flag_raw is not None else ""
        is_top = top_flag in {"是", "Y", "YES", "True", "true", "1", "√"}

        index[key] = {
            "cas_major_category": str(major_category).strip() if major_category is not None else None,
            "cas_major_tier": major_tier,
            "cas_small_category": str(small_category).strip() if small_category is not None else None,
            "cas_small_tier": small_tier,
            "cas_top": 1 if is_top else 0,
            "cas_year": table_name.replace("FQBJCR", ""),
        }

    return index


def match_papers_with_jcr(papers: list[dict], jcr_index: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for row in papers:
        pmid = str(row.get("pmid", "")).strip()
        journal = str(row.get("journal", "")).strip()
        if not pmid or not journal:
            continue

        key = _normalize_journal_name(journal)
        matched = jcr_index.get(key)
        if not matched:
            continue

        out.append(
            {
                "pmid": pmid,
                "jcr_if": matched.get("jcr_if"),
                "jcr_quartile": matched.get("jcr_quartile"),
                "jcr_year": matched.get("jcr_year"),
            }
        )
    return out


def match_papers_with_cas(papers: list[dict], cas_index: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for row in papers:
        pmid = str(row.get("pmid", "")).strip()
        journal = str(row.get("journal", "")).strip()
        if not pmid or not journal:
            continue

        key = _normalize_journal_name(journal)
        matched = cas_index.get(key)
        if not matched:
            continue

        out.append(
            {
                "pmid": pmid,
                "cas_major_category": matched.get("cas_major_category"),
                "cas_major_tier": matched.get("cas_major_tier"),
                "cas_small_category": matched.get("cas_small_category"),
                "cas_small_tier": matched.get("cas_small_tier"),
                "cas_top": matched.get("cas_top"),
                "cas_year": matched.get("cas_year"),
            }
        )
    return out
