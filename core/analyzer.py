from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd


STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "among",
    "effect",
    "effects",
    "study",
    "clinical",
    "patients",
    "patient",
    "based",
    "using",
    "analysis",
    "risk",
    "association",
    "review",
    "meta",
    "trial",
    "outcomes",
    "outcome",
}

INTERVENTION_TERMS = [
    "therapy",
    "treatment",
    "drug",
    "medication",
    "surgery",
    "stent",
    "ventilation",
    "vaccination",
    "supplement",
    "rehabilitation",
    "telemedicine",
    "monitoring",
]

OUTCOME_TERMS = [
    "mortality",
    "survival",
    "readmission",
    "complication",
    "adverse",
    "quality of life",
    "hospitalization",
    "efficacy",
    "safety",
    "response",
    "recurrence",
]

POPULATION_TERMS = [
    "elderly",
    "children",
    "pediatric",
    "pregnancy",
    "women",
    "men",
    "icu",
    "critical care",
    "outpatient",
    "primary care",
    "asian",
    "african",
]


@dataclass
class TrendResult:
    yearly_counts: pd.DataFrame
    top_keywords: pd.DataFrame
    growth_keywords: pd.DataFrame


def _split_semicolon(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def _to_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            str(row.get("mesh_terms", "")),
            str(row.get("keywords", "")),
            str(row.get("publication_types", "")),
        ]
    ).lower()


def extract_top_terms(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    text = " ".join((df["title"].fillna("") + " " + df["abstract"].fillna(""))).lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens).most_common(top_n)
    return pd.DataFrame(counts, columns=["term", "count"])


def compute_trends(df: pd.DataFrame) -> TrendResult:
    work = df.copy()
    work["pub_year_num"] = pd.to_numeric(work["pub_year"], errors="coerce")
    work = work.dropna(subset=["pub_year_num"])  # type: ignore[arg-type]

    yearly_counts = (
        work.groupby("pub_year_num", as_index=False)
        .size()
        .rename(columns={"pub_year_num": "year", "size": "papers"})
        .sort_values("year")
    )

    top_keywords = extract_top_terms(work, top_n=30)

    if yearly_counts.empty:
        growth_keywords = pd.DataFrame(columns=["term", "recent_count", "past_count", "growth_ratio"])
    else:
        max_year = int(yearly_counts["year"].max())
        recent_years = {max_year, max_year - 1}

        recent_text = " ".join(
            (
                work[work["pub_year_num"].isin(recent_years)]["title"].fillna("")
                + " "
                + work[work["pub_year_num"].isin(recent_years)]["abstract"].fillna("")
            )
        ).lower()
        past_text = " ".join(
            (
                work[~work["pub_year_num"].isin(recent_years)]["title"].fillna("")
                + " "
                + work[~work["pub_year_num"].isin(recent_years)]["abstract"].fillna("")
            )
        ).lower()

        recent_tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", recent_text) if t not in STOPWORDS]
        past_tokens = [t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", past_text) if t not in STOPWORDS]

        rc = Counter(recent_tokens)
        pc = Counter(past_tokens)
        rows = []
        for term, recent_count in rc.most_common(80):
            if recent_count < 3:
                continue
            past_count = pc.get(term, 0)
            growth_ratio = (recent_count + 1) / (past_count + 1)
            rows.append((term, recent_count, past_count, round(growth_ratio, 2)))

        growth_keywords = (
            pd.DataFrame(rows, columns=["term", "recent_count", "past_count", "growth_ratio"])
            .sort_values(["growth_ratio", "recent_count"], ascending=False)
            .head(20)
        )

    return TrendResult(
        yearly_counts=yearly_counts,
        top_keywords=top_keywords,
        growth_keywords=growth_keywords,
    )


def evidence_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = df.copy()
    data["text"] = data.apply(_to_text, axis=1)

    def count_matches(terms: list[str], label: str) -> pd.DataFrame:
        rows = []
        for term in terms:
            matched = data[data["text"].str.contains(re.escape(term), case=False, na=False)]
            if len(matched) > 0:
                rows.append((term, len(matched), label))
        return pd.DataFrame(rows, columns=["item", "count", "group"])

    intervention_df = count_matches(INTERVENTION_TERMS, "intervention")
    outcome_df = count_matches(OUTCOME_TERMS, "outcome")
    population_df = count_matches(POPULATION_TERMS, "population")

    study_rows = []
    for value in data["publication_types"].fillna(""):
        for item in _split_semicolon(value):
            study_rows.append(item)
    study_design_df = (
        pd.DataFrame(Counter(study_rows).most_common(20), columns=["item", "count"])
        if study_rows
        else pd.DataFrame(columns=["item", "count"])
    )
    if not study_design_df.empty:
        study_design_df["group"] = "study_design"

    disease_rows = []
    for value in data["mesh_terms"].fillna(""):
        for item in _split_semicolon(value)[:5]:
            disease_rows.append(item)
    disease_df = (
        pd.DataFrame(Counter(disease_rows).most_common(20), columns=["item", "count"])
        if disease_rows
        else pd.DataFrame(columns=["item", "count"])
    )
    if not disease_df.empty:
        disease_df["group"] = "disease"

    return {
        "disease": disease_df,
        "intervention": intervention_df.sort_values("count", ascending=False).head(20),
        "outcome": outcome_df.sort_values("count", ascending=False).head(20),
        "population": population_df.sort_values("count", ascending=False).head(20),
        "study_design": study_design_df,
    }
