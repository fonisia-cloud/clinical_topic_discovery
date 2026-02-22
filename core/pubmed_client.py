from __future__ import annotations

import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests


class PubMedClient:
    def __init__(self, email: str = "", api_key: str = "", tool: str = "clinical-topic-discovery") -> None:
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.openalex_base_url = "https://api.openalex.org"
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.session = requests.Session()

    def _params(self, **kwargs: str | int) -> dict:
        params = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(kwargs)
        return params

    def _get(self, endpoint: str, params: dict, retries: int = 3) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1.2 * (attempt + 1))
        raise RuntimeError(f"PubMed request failed for {endpoint}: {last_error}")

    def search_pmids(self, query: str, retmax: int = 200) -> dict:
        response = self._get(
            "esearch.fcgi",
            self._params(db="pubmed", term=query, retmode="json", retmax=retmax, sort="relevance"),
        )
        payload = response.json().get("esearchresult", {})
        return {
            "idlist": payload.get("idlist", []),
            "count": int(payload.get("count", 0)),
        }

    def fetch_summaries(self, pmids: list[str]) -> dict[str, dict]:
        if not pmids:
            return {}

        summaries: dict[str, dict] = {}
        for i in range(0, len(pmids), 200):
            batch = pmids[i : i + 200]
            response = self._get(
                "esummary.fcgi",
                self._params(db="pubmed", id=",".join(batch), retmode="json", version="2.0"),
            )
            result = response.json().get("result", {})
            for pmid in batch:
                item = result.get(pmid)
                if item:
                    summaries[pmid] = item
            time.sleep(0.35)
        return summaries

    def fetch_abstract_metadata(self, pmids: list[str]) -> dict[str, dict]:
        if not pmids:
            return {}

        data: dict[str, dict] = {}
        for i in range(0, len(pmids), 100):
            batch = pmids[i : i + 100]
            response = self._get(
                "efetch.fcgi",
                self._params(db="pubmed", id=",".join(batch), rettype="abstract", retmode="xml"),
            )

            root = ET.fromstring(response.text)
            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID")
                if not pmid:
                    continue

                abstract_parts: list[str] = []
                for part in article.findall(".//Abstract/AbstractText"):
                    label = part.attrib.get("Label", "").strip()
                    text = "".join(part.itertext()).strip()
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)

                mesh_terms = [
                    node.text.strip()
                    for node in article.findall(".//MeshHeading/DescriptorName")
                    if node.text and node.text.strip()
                ]

                publication_types = [
                    node.text.strip()
                    for node in article.findall(".//PublicationTypeList/PublicationType")
                    if node.text and node.text.strip()
                ]

                keywords = [
                    node.text.strip()
                    for node in article.findall(".//KeywordList/Keyword")
                    if node.text and node.text.strip()
                ]

                data[pmid] = {
                    "abstract": " ".join(abstract_parts),
                    "mesh_terms": mesh_terms,
                    "publication_types": publication_types,
                    "keywords": keywords,
                }
            time.sleep(0.35)
        return data

    def fetch_openalex_metrics(self, pmid_doi_rows: list[dict]) -> list[dict]:
        metrics: list[dict] = []

        for row in pmid_doi_rows:
            pmid = str(row.get("pmid", "")).strip()
            doi = str(row.get("doi", "")).strip()
            if not pmid or not doi:
                continue

            doi_normalized = doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()
            if not doi_normalized:
                continue

            encoded_doi = urllib.parse.quote(f"https://doi.org/{doi_normalized}", safe="")
            url = f"{self.openalex_base_url}/works/{encoded_doi}"

            try:
                response = self.session.get(url, timeout=20)
                if response.status_code != 200:
                    continue
                payload = response.json()
            except Exception:  # noqa: BLE001
                continue

            cited_by_count = payload.get("cited_by_count")
            journal_impact_score = None
            source = payload.get("primary_location", {}).get("source", {}) if isinstance(payload, dict) else {}
            summary_stats = source.get("summary_stats", {}) if isinstance(source, dict) else {}
            if isinstance(summary_stats, dict):
                journal_impact_score = summary_stats.get("2yr_mean_citedness")

            metrics.append(
                {
                    "pmid": pmid,
                    "citation_count": int(cited_by_count) if cited_by_count is not None else None,
                    "journal_impact_score": float(journal_impact_score) if journal_impact_score is not None else None,
                    "impact_source": "openalex",
                }
            )
            time.sleep(0.08)

        return metrics

    @staticmethod
    def merge_records(pmids: list[str], summaries: Dict[str, dict], abstracts: Dict[str, dict]) -> List[dict]:
        rows: list[dict] = []
        for pmid in pmids:
            summary = summaries.get(pmid, {})
            abstract_data = abstracts.get(pmid, {})

            doi = ""
            for article_id in summary.get("articleids", []):
                if article_id.get("idtype") == "doi":
                    doi = article_id.get("value", "")
                    break

            author_names = [a.get("name", "") for a in summary.get("authors", []) if a.get("name")]

            rows.append(
                {
                    "pmid": pmid,
                    "title": summary.get("title", "").strip("[] "),
                    "abstract": abstract_data.get("abstract", ""),
                    "journal": summary.get("fulljournalname", "") or summary.get("source", ""),
                    "pub_date": summary.get("pubdate", ""),
                    "pub_year": str(summary.get("pubdate", ""))[:4],
                    "doi": doi,
                    "authors": "; ".join(author_names),
                    "publication_types": "; ".join(abstract_data.get("publication_types", [])),
                    "mesh_terms": "; ".join(abstract_data.get("mesh_terms", [])),
                    "keywords": "; ".join(abstract_data.get("keywords", [])),
                    "citation_count": None,
                    "journal_impact_score": None,
                    "impact_source": None,
                    "jcr_if": None,
                    "jcr_quartile": None,
                    "jcr_year": None,
                    "cas_major_category": None,
                    "cas_major_tier": None,
                    "cas_small_category": None,
                    "cas_small_tier": None,
                    "cas_top": None,
                    "cas_year": None,
                }
            )
        return rows
