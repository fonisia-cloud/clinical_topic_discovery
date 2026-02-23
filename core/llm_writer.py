from __future__ import annotations

import json
import re

import requests


def _candidate_chat_urls(api_base: str) -> list[str]:
    base = api_base.strip().rstrip("/")
    if not base:
        return []

    if base.endswith("/chat/completions"):
        return [base]

    candidates = [
        f"{base}/chat/completions",
        f"{base}/v1/chat/completions",
    ]

    seen = set()
    out = []
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _extract_openai_content(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    text_parts.append(str(text))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts).strip()

    return str(content).strip() if content else ""


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    if not api_base or not api_key or not model_name:
        raise RuntimeError("Missing API base URL, API key, or model name.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a senior medical research advisor."},
            {"role": "user", "content": prompt},
        ],
    }

    errors: list[str] = []
    for url in _candidate_chat_urls(api_base):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if response.status_code == 404:
                errors.append(f"404 at {url}")
                continue
            response.raise_for_status()
            data = response.json()
            content = _extract_openai_content(data)
            if not content:
                raise RuntimeError(f"Model returned empty content from {url}.")
            return content
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")

    raise RuntimeError("API request failed. Tried endpoints: " + " | ".join(errors))


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return stripped


def build_topic_ideas_prompt(
    query_text: str,
    rule_topics: list[dict],
    evidence_records: list[dict],
    top_n: int = 5,
) -> str:
    topic_lines = []
    for idx, item in enumerate(rule_topics[: max(1, top_n * 2)]):
        topic_lines.append(
            f"{idx + 1}. title={item.get('topic_title')}; score={item.get('score')}; "
            f"rationale={item.get('rationale')}; gap={item.get('key_gap')}; pmids={','.join(item.get('supporting_pmids', []))}"
        )

    evidence_lines = []
    for row in evidence_records[:120]:
        evidence_lines.append(
            f"PMID {row.get('pmid')}: {row.get('title')} ({row.get('pub_year')}, {row.get('journal')}); "
            f"type={row.get('publication_types', '')}; abstract={str(row.get('abstract', ''))[:500]}"
        )

    return (
        "You are a senior medical research advisor.\n"
        "Task: generate final clinical topic ideas for paper planning based on evidence and preliminary rule-engine candidates.\n"
        "Requirements:\n"
        "1) Be evidence-grounded and conservative.\n"
        "2) Do not fabricate PMIDs or studies.\n"
        "3) Each topic must include why it matters and what gap remains.\n"
        "4) Prefer clinically meaningful and feasible directions.\n"
        f"5) Return exactly top {top_n} topics in JSON array format only.\n\n"
        "Output JSON schema per item:\n"
        "{\n"
        "  \"topic_title\": string,\n"
        "  \"rationale\": string,\n"
        "  \"key_gap\": string,\n"
        "  \"score\": number,\n"
        "  \"supporting_pmids\": [string]\n"
        "}\n\n"
        f"User query: {query_text}\n\n"
        "Rule-engine candidates:\n"
        + "\n".join(topic_lines)
        + "\n\nEvidence set:\n"
        + "\n".join(evidence_lines)
    )


def enhance_topic_ideas_with_llm(
    api_base: str,
    api_key: str,
    model_name: str,
    query_text: str,
    rule_topics: list[dict],
    evidence_records: list[dict],
    top_n: int = 5,
    temperature: float = 0.1,
) -> list[dict]:
    prompt = build_topic_ideas_prompt(
        query_text=query_text,
        rule_topics=rule_topics,
        evidence_records=evidence_records,
        top_n=top_n,
    )
    raw = call_openai_compatible(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
    )

    json_block = _extract_json_block(raw)
    data = json.loads(json_block)
    if not isinstance(data, list):
        raise RuntimeError("LLM topic output is not a JSON array.")

    out: list[dict] = []
    for item in data[:top_n]:
        if not isinstance(item, dict):
            continue
        pmids = item.get("supporting_pmids", [])
        if not isinstance(pmids, list):
            pmids = []
        out.append(
            {
                "topic_title": str(item.get("topic_title", "")).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
                "key_gap": str(item.get("key_gap", "")).strip(),
                "score": float(item.get("score", 0) or 0),
                "supporting_pmids": [str(p).strip() for p in pmids if str(p).strip()],
            }
        )

    out = [x for x in out if x["topic_title"]]
    if not out:
        raise RuntimeError("LLM returned empty topic list.")
    return out
