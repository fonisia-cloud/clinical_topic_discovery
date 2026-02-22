from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .analyzer import INTERVENTION_TERMS, OUTCOME_TERMS, POPULATION_TERMS, compute_trends


DEFAULT_SCORE_WEIGHTS = {
    "clinical_value": 0.4,
    "innovation": 0.35,
    "feasibility": 0.25,
}

GENERIC_MESH_TERMS = {
    "Humans",
    "Male",
    "Female",
    "Adult",
    "Aged",
    "Middle Aged",
    "Young Adult",
    "Child",
    "Adolescent",
    "Infant",
    "Retrospective Studies",
    "Prospective Studies",
}


def _contains_any(text: str, candidates: list[str]) -> list[str]:
    lower = text.lower()
    return [item for item in candidates if item in lower]


def _extract_focus_terms(df: pd.DataFrame, col: str, top_n: int = 20, exclude: set[str] | None = None) -> list[str]:
    terms: list[str] = []
    exclude = exclude or set()

    for value in df[col].fillna(""):
        for item in str(value).split(";"):
            item = item.strip()
            if not item:
                continue
            if item in exclude:
                continue
            if len(item) <= 2:
                continue
            terms.append(item)
    return [t for t, _ in Counter(terms).most_common(top_n)]


def _normalize_weights(score_weights: dict[str, float] | None) -> dict[str, float]:
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if score_weights:
        for key in weights:
            value = score_weights.get(key)
            if value is not None and value >= 0:
                weights[key] = float(value)

    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_SCORE_WEIGHTS)

    return {k: v / total for k, v in weights.items()}


def _minmax_norm(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    smin = float(series.min())
    smax = float(series.max())
    if smax == smin:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin) * 100


def _extract_query_concepts(query_text: str) -> list[str]:
    if not query_text:
        return []

    # Remove field tags, operators and punctuation while keeping useful terms.
    cleaned = re.sub(r"\[[^\]]+\]", " ", query_text)
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("(", " ").replace(")", " ").replace('"', " ").replace(",", " ")

    words = [w.strip().lower() for w in cleaned.split() if w.strip()]
    stop = {"the", "and", "for", "with", "from", "study", "clinical"}
    phrases: list[str] = []

    # Keep both unigram and nearby bi-gram concepts for query alignment.
    for word in words:
        if len(word) >= 4 and word not in stop:
            phrases.append(word)

    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if len(a) >= 3 and len(b) >= 3 and a not in stop and b not in stop:
            phrases.append(f"{a} {b}")

    return list(dict.fromkeys(phrases))[:20]


def _query_alignment_score(text: str, query_concepts: list[str]) -> float:
    if not query_concepts:
        return 0.0

    matched = 0
    lowered = text.lower()
    for concept in query_concepts:
        if concept in lowered:
            matched += 1

    return matched / len(query_concepts) * 100


def generate_topic_candidates(
    df: pd.DataFrame,
    top_n: int = 5,
    score_weights: dict[str, float] | None = None,
    query_text: str = "",
) -> list[dict]:
    if df.empty:
        return []

    weights = _normalize_weights(score_weights)
    query_concepts = _extract_query_concepts(query_text)

    work = df.copy()
    work["text"] = (
        work["title"].fillna("")
        + " "
        + work["abstract"].fillna("")
        + " "
        + work["mesh_terms"].fillna("")
        + " "
        + work["keywords"].fillna("")
    ).str.lower()
    work["pub_year_num"] = pd.to_numeric(work["pub_year"], errors="coerce")

    mesh_candidates = _extract_focus_terms(work, "mesh_terms", top_n=40, exclude=GENERIC_MESH_TERMS)
    keyword_candidates = _extract_focus_terms(work, "keywords", top_n=30)

    trends = compute_trends(work)
    growth_terms = set(trends.growth_keywords["term"].head(30).tolist()) if not trends.growth_keywords.empty else set()

    candidate_terms: list[str] = []
    candidate_terms.extend(mesh_candidates)
    candidate_terms.extend(keyword_candidates)

    # Add query concepts so results remain query-aware even when MeSH is sparse.
    for concept in query_concepts:
        if concept not in candidate_terms and len(concept) >= 4:
            candidate_terms.append(concept)

    # Keep unique while preserving order.
    seen = set()
    ordered_candidates = []
    for item in candidate_terms:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(item)

    suggestions: list[dict] = []

    for candidate in ordered_candidates[:30]:
        escaped = re.escape(candidate.lower())
        candidate_match = work[work["text"].str.contains(escaped, na=False, regex=True)]
        if len(candidate_match) < 5:
            continue

        sample_text = " ".join(candidate_match["text"].head(200).tolist())

        interventions = _contains_any(sample_text, INTERVENTION_TERMS)
        outcomes = _contains_any(sample_text, OUTCOME_TERMS)
        populations = _contains_any(sample_text, POPULATION_TERMS)

        intervention = interventions[0] if interventions else "targeted intervention"
        outcome = outcomes[0] if outcomes else "clinical outcomes"
        population = populations[0] if populations else "clinical population"

        recent_count = len(candidate_match[candidate_match["pub_year_num"] >= candidate_match["pub_year_num"].max() - 1])
        historical_count = max(len(candidate_match) - recent_count, 1)
        rct_count = int(candidate_match["publication_types"].fillna("").str.contains("Randomized", case=False).sum())

        publication_type_diversity = int(
            candidate_match["publication_types"]
            .fillna("")
            .str.split(";")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

        supporting_pmids = candidate_match["pmid"].head(12).astype(str).tolist()
        abstract_coverage = float(candidate_match["abstract"].fillna("").str.len().gt(50).mean())

        growth_presence = 1 if any(token in sample_text for token in growth_terms) else 0
        recent_ratio = recent_count / max(len(candidate_match), 1)
        query_alignment = _query_alignment_score(sample_text, query_concepts)

        clinical_raw = (
            recent_ratio * 35
            + min(len(candidate_match), 80) / 80 * 30
            + (20 if outcomes else 8)
            + query_alignment * 0.15
        )

        innovation_raw = (
            (42 if rct_count <= 1 else 28 if rct_count <= 3 else 10)
            + (26 if growth_presence else 8)
            + (14 if publication_type_diversity <= 3 else 6)
            + (10 if populations else 4)
            + query_alignment * 0.2
        )

        feasibility_raw = (
            min(len(candidate_match), 80) / 80 * 50
            + min(len(supporting_pmids), 12) / 12 * 25
            + (20 if abstract_coverage >= 0.7 else 12 if abstract_coverage >= 0.4 else 5)
            + query_alignment * 0.1
        )

        suggestions.append(
            {
                "topic_title": f"Association between {intervention} and {outcome} in {candidate} ({population})",
                "rationale": (
                    f"{candidate} has {len(candidate_match)} related papers with {recent_count} in the recent two-year window. "
                    f"Query alignment score is {query_alignment:.1f}."
                ),
                "key_gap": (
                    f"Randomized evidence signal is {rct_count}. Study design diversity is {publication_type_diversity}. "
                    "This suggests room for pragmatic clinical-data studies and external validation."
                ),
                "supporting_pmids": supporting_pmids,
                "raw_clinical": float(clinical_raw),
                "raw_innovation": float(innovation_raw),
                "raw_feasibility": float(feasibility_raw),
            }
        )

    if not suggestions:
        return []

    score_df = pd.DataFrame(suggestions)
    score_df["component_clinical_value"] = _minmax_norm(score_df["raw_clinical"])
    score_df["component_innovation"] = _minmax_norm(score_df["raw_innovation"])
    score_df["component_feasibility"] = _minmax_norm(score_df["raw_feasibility"])

    score_df["score"] = (
        score_df["component_clinical_value"] * weights["clinical_value"]
        + score_df["component_innovation"] * weights["innovation"]
        + score_df["component_feasibility"] * weights["feasibility"]
    ).round(2)

    score_df = score_df.sort_values("score", ascending=False)

    deduped: list[dict] = []
    seen_titles: set[str] = set()
    for _, row in score_df.iterrows():
        title = str(row["topic_title"])
        if title in seen_titles:
            continue
        deduped.append(
            {
                "topic_title": title,
                "rationale": str(row["rationale"]),
                "key_gap": str(row["key_gap"]),
                "score": float(row["score"]),
                "component_clinical_value": round(float(row["component_clinical_value"]), 2),
                "component_innovation": round(float(row["component_innovation"]), 2),
                "component_feasibility": round(float(row["component_feasibility"]), 2),
                "supporting_pmids": list(row["supporting_pmids"]),
            }
        )
        seen_titles.add(title)
        if len(deduped) >= top_n:
            break

    return deduped
