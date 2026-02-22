from __future__ import annotations

import re
from typing import Iterable, List


ARTICLE_TYPE_MAP = {
    "Randomized Controlled Trial": '"randomized controlled trial"[Publication Type]',
    "Systematic Review": '"systematic review"[Publication Type]',
    "Meta-Analysis": '"meta-analysis"[Publication Type]',
    "Observational Study": '"observational study"[Publication Type]',
    "Cohort Study": '"cohort studies"[MeSH Terms]',
    "Case-Control Study": '"case-control studies"[MeSH Terms]',
    "Clinical Trial": '"clinical trial"[Publication Type]',
}


BOOLEAN_TOKEN_PATTERN = re.compile(r'("[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b)', flags=re.IGNORECASE)


def _clean_parts(parts: Iterable[str]) -> List[str]:
    return [part.strip() for part in parts if part and part.strip()]


def _has_pubmed_field(term: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]", term))


def _format_term(term: str, default_field: str, query_mode: str) -> str:
    clean = term.strip()
    if not clean:
        return ""

    if _has_pubmed_field(clean):
        return clean

    if query_mode == "pubmed_default":
        # Keep term untagged so PubMed can apply Automatic Term Mapping, same as website default behavior.
        return clean

    if clean.startswith('"') and clean.endswith('"'):
        return f"{clean}[{default_field}]"

    if " " in clean:
        return f'"{clean}"[{default_field}]'

    return f"{clean}[{default_field}]"


def _build_field_expression(raw_text: str, default_field: str, query_mode: str) -> str:
    if not raw_text.strip():
        return ""

    # Keep backward compatibility: comma-separated terms are treated as OR.
    normalized = raw_text.replace(",", " OR ")
    parts = [part.strip() for part in BOOLEAN_TOKEN_PATTERN.split(normalized) if part and part.strip()]
    if not parts:
        return ""

    rendered: list[str] = []
    for part in parts:
        upper = part.upper()
        if upper in {"AND", "OR", "NOT"}:
            rendered.append(upper)
        elif part in {"(", ")"}:
            rendered.append(part)
        else:
            rendered.append(_format_term(part, default_field, query_mode))

    return f"({' '.join(rendered)})"


def build_query(
    keywords: str,
    mesh_terms: str = "",
    start_year: int | None = None,
    end_year: int | None = None,
    article_types: list[str] | None = None,
    keyword_mode: str = "pubmed_default",
) -> str:
    clauses: list[str] = []

    keyword_clause = _build_field_expression(keywords, "Title/Abstract", keyword_mode)
    if keyword_clause:
        clauses.append(keyword_clause)

    mesh_clause = _build_field_expression(mesh_terms, "MeSH Terms", "field_restricted")
    if mesh_clause:
        clauses.append(mesh_clause)

    if start_year and end_year and start_year <= end_year:
        clauses.append(f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])')

    selected_types = [ARTICLE_TYPE_MAP[t] for t in (article_types or []) if t in ARTICLE_TYPE_MAP]
    if selected_types:
        clauses.append(f"({' OR '.join(selected_types)})")

    if not clauses:
        return "all[sb]"

    return " AND ".join(clauses)
